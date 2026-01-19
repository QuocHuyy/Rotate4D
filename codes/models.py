import os
import logging
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, ModeType, TestDataset


class KGEModel(nn.Module, ABC):
    """
    Abstract base class for Knowledge Graph Embedding models.
    
    Subclasses must define:
        - self.entity_embedding: nn.Parameter for entity embeddings
        - self.relation_embedding: nn.Parameter for relation embeddings
        - func(): scoring function for triples
    """

    @abstractmethod
    def func(
        self, 
        head: torch.Tensor, 
        rel: torch.Tensor, 
        tail: torch.Tensor, 
        batch_type: BatchType
    ) -> torch.Tensor:
        """
        Scoring function for knowledge graph triples.
        
        Args:
            head: Head entity embeddings
                - SINGLE: [batch_size, 1, hidden_dim]
                - HEAD_BATCH: [batch_size, neg_size, hidden_dim]
                - TAIL_BATCH: [batch_size, 1, hidden_dim]
            rel: Relation embeddings [batch_size, 1, hidden_dim]
            tail: Tail entity embeddings
                - SINGLE: [batch_size, 1, hidden_dim]
                - HEAD_BATCH: [batch_size, 1, hidden_dim]
                - TAIL_BATCH: [batch_size, neg_size, hidden_dim]
            batch_type: Type of batch (SINGLE, HEAD_BATCH, or TAIL_BATCH)
            
        Returns:
            Scores for the triples [batch_size, num_samples]
        """
        raise NotImplementedError

    def forward(
        self, 
        sample: torch.Tensor, 
        batch_type: BatchType = BatchType.SINGLE
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: extract embeddings and compute scores.

        Args:
            sample: Input samples with different formats per batch type:
                - SINGLE: [batch_size, 3] tensor of (head, rel, tail) indices
                - HEAD_BATCH/TAIL_BATCH: tuple of (positive_sample, negative_sample)
                    - positive_sample: [batch_size, 3]
                    - negative_sample: [batch_size, neg_sample_size]
            batch_type: Type of batch
                - SINGLE: for positive samples and validation/testing
                - HEAD_BATCH: for (?, r, t) tasks in training
                - TAIL_BATCH: for (h, r, ?) tasks in training

        Returns:
            scores: Computed scores for the samples
            entities: Tuple of (head_embeddings, tail_embeddings) for regularization
        """
        if batch_type == BatchType.SINGLE:
            head = self._select_embeddings(
                self.entity_embedding, 
                sample[:, 0]
            ).unsqueeze(1)
            
            relation = self._select_embeddings(
                self.relation_embedding, 
                sample[:, 1]
            ).unsqueeze(1)
            
            tail = self._select_embeddings(
                self.entity_embedding, 
                sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, neg_sample_size = head_part.shape
            
            head = self._select_embeddings(
                self.entity_embedding,
                head_part.view(-1)
            ).view(batch_size, neg_sample_size, -1)
            
            relation = self._select_embeddings(
                self.relation_embedding,
                tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = self._select_embeddings(
                self.entity_embedding,
                tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, neg_sample_size = tail_part.shape
            
            head = self._select_embeddings(
                self.entity_embedding,
                head_part[:, 0]
            ).unsqueeze(1)
            
            relation = self._select_embeddings(
                self.relation_embedding,
                head_part[:, 1]
            ).unsqueeze(1)
            
            tail = self._select_embeddings(
                self.entity_embedding,
                tail_part.view(-1)
            ).view(batch_size, neg_sample_size, -1)

        else:
            raise ValueError(f'Batch type {batch_type} not supported!')

        score = self.func(head, relation, tail, batch_type)
        return score, (head, tail)

    @staticmethod
    def _select_embeddings(
        embedding: nn.Parameter, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Select embeddings by indices.
        
        Args:
            embedding: Embedding matrix [num_items, embedding_dim]
            indices: Indices to select [batch_size] or [batch_size * k]
            
        Returns:
            Selected embeddings
        """
        return torch.index_select(embedding, dim=0, index=indices)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        Execute a single training step.
        
        Args:
            model: KGEModel instance
            optimizer: PyTorch optimizer
            train_iterator: Iterator over training data
            args: Training arguments
            
        Returns:
            Dictionary of training metrics
        """
        model.train()
        optimizer.zero_grad()

        # Get batch data
        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)
        
        # Move to GPU
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # Compute negative scores with adversarial sampling
        negative_score, _ = model(
            (positive_sample, negative_sample), 
            batch_type=batch_type
        )
        
        if args.use_adversarial_negative_sampling:
            # Self-adversarial negative sampling
            negative_score = (
                F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                * F.logsigmoid(-negative_score)
            ).sum(dim=1)
        else:
            # Uniform negative sampling
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        # Compute positive scores
        positive_score, entities = model(positive_sample, batch_type=BatchType.SINGLE)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        # Compute loss with subsampling weight
        positive_sample_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        
        loss = (positive_sample_loss + negative_sample_loss) / 2

        # Add regularization if enabled
        if args.regularization > 0:
            head_emb, tail_emb = entities
            regularization = args.regularization * (
                head_emb.norm(p=2) ** 2 + 
                tail_emb.norm(p=2) ** 2
            ) / head_emb.shape[0]
            loss = loss + regularization
        else:
            regularization = torch.tensor(0.0)

        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping if enabled
        if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()

        # Return metrics
        return {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'regularization': regularization.item()
        }

    @staticmethod
    def test_step(
        model, 
        data_reader, 
        mode: ModeType, 
        args
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        """
        Evaluate the model on validation or test datasets.
        
        Args:
            model: KGEModel instance
            data_reader: DataReader instance
            mode: ModeType.VALID or ModeType.TEST
            args: Evaluation arguments
            
        Returns:
            metrics: Overall metrics (MRR, MR, HITS@k)
            metrics_per_relation: Metrics broken down by relation
        """
        model.eval()

        # Prepare dataloaders for both head and tail prediction
        test_dataloader_head = DataLoader(
            TestDataset(data_reader, mode, BatchType.HEAD_BATCH),
            batch_size=args.test_batch_size,
            num_workers=max(0, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(data_reader, mode, BatchType.TAIL_BATCH),
            batch_size=args.test_batch_size,
            num_workers=max(0, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        # Initialize logging structures
        logs = []
        logs_per_relation = defaultdict(list)

        step = 0
        total_steps = sum(len(dataset) for dataset in test_dataset_list)

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    # Move to GPU
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    # Compute scores
                    score, _ = model((positive_sample, negative_sample), batch_type)
                    
                    # Apply filter bias (mask out true triples)
                    score = score + filter_bias

                    # Sort entities by score (descending)
                    argsort = torch.argsort(score, dim=1, descending=True)

                    # Get the correct entity index
                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError(f'Batch type {batch_type} not supported')

                    # Compute metrics for each sample in batch
                    for i in range(batch_size):
                        # Find ranking of the correct entity
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero(as_tuple=False)
                        assert ranking.size(0) == 1, "Each entity should appear exactly once"

                        ranking = ranking.item() + 1  # Convert to 1-based ranking
                        relation_id = positive_sample[i, 1].item()

                        # Compute evaluation metrics
                        sample_metrics = {
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        }
                        
                        logs.append(sample_metrics)
                        logs_per_relation[relation_id].append(sample_metrics)

                    # Log progress
                    if step % args.test_log_steps == 0:
                        logging.info(f'Evaluating the model... ({step}/{total_steps})')

                    step += 1

        # Aggregate overall metrics
        metrics = {}
        for metric_name in logs[0].keys():
            metrics[metric_name] = sum(log[metric_name] for log in logs) / len(logs)

        # Aggregate per-relation metrics
        metrics_per_relation = {}
        for rel_id, rel_logs in logs_per_relation.items():
            metrics_per_relation[rel_id] = {}
            for metric_name in rel_logs[0].keys():
                metrics_per_relation[rel_id][metric_name] = (
                    sum(log[metric_name] for log in rel_logs) / len(rel_logs)
                )

        return metrics, metrics_per_relation


class Rotate3D(KGEModel):
    """
    Rotate3D model: represents relations as rotations in 3D space.
    
    Uses spherical coordinates to represent rotation axis and angle.
    """
    
    def __init__(
        self, 
        num_entity: int, 
        num_relation: int, 
        hidden_dim: int, 
        gamma: float, 
        p_norm: int = 2
    ):
        """
        Initialize Rotate3D model.
        
        Args:
            num_entity: Number of entities
            num_relation: Number of relations
            hidden_dim: Dimension of embeddings
            gamma: Margin parameter
            p_norm: P-norm for distance calculation
        """
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm
        self.pi = math.pi

        # Margin parameter (fixed)
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        # Embedding range for initialization
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        # Entity embeddings: (i, j, k) components
        self.entity_embedding = nn.Parameter(
            torch.zeros(num_entity, hidden_dim * 3)
        )
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Relation embeddings: (beta_1, beta_2, theta, bias)
        # beta_1, beta_2: spherical coordinates for rotation axis
        # theta: rotation angle
        # bias: scaling factor
        self.relation_embedding = nn.Parameter(
            torch.zeros(num_relation, hidden_dim * 4)
        )
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Initialize bias component to 1
        nn.init.ones_(
            tensor=self.relation_embedding[:, 3*hidden_dim:4*hidden_dim]
        )

    def func(
        self, 
        head: torch.Tensor, 
        rel: torch.Tensor, 
        tail: torch.Tensor, 
        batch_type: BatchType
    ) -> torch.Tensor:
        """
        Scoring function using 3D rotations.
        
        Implements rotation of head entity in 3D space using Rodrigues' rotation formula.
        """
        # Decompose entity embeddings into 3D components
        head_i, head_j, head_k = torch.chunk(head, 3, dim=2)
        tail_i, tail_j, tail_k = torch.chunk(tail, 3, dim=2)
        
        # Decompose relation embeddings
        beta_1, beta_2, theta, bias = torch.chunk(rel, 4, dim=2)
        bias = torch.abs(bias)  # Ensure positive scaling

        # Normalize angles to [-pi, pi]
        beta_1 = beta_1 / (self.embedding_range.item() / self.pi)
        beta_2 = beta_2 / (self.embedding_range.item() / self.pi)
        theta = theta / (self.embedding_range.item() / self.pi)

        # Precompute trigonometric values
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Compute rotation axis using spherical coordinates
        # (beta_1, beta_2) -> (rel_i, rel_j, rel_k)
        rel_i = torch.cos(beta_1)
        rel_j = torch.sin(beta_1) * torch.cos(beta_2)
        rel_k = torch.sin(beta_1) * torch.sin(beta_2)

        # Apply Rodrigues' rotation formula
        # v_rot = v*cos(θ) + (k×v)*sin(θ) + k(k·v)(1-cos(θ))
        C = rel_i * head_i + rel_j * head_j + rel_k * head_k
        C = C * (1 - cos_theta)

        # Compute rotated head entity
        new_head_i = (
            head_i * cos_theta + 
            C * rel_i + 
            sin_theta * (rel_j * head_k - head_j * rel_k)
        )
        new_head_j = (
            head_j * cos_theta + 
            C * rel_j - 
            sin_theta * (rel_i * head_k - head_i * rel_k)
        )
        new_head_k = (
            head_k * cos_theta + 
            C * rel_k + 
            sin_theta * (rel_i * head_j - head_i * rel_j)
        )

        # Compute distance scores
        score_i = new_head_i * bias - tail_i
        score_j = new_head_j * bias - tail_j
        score_k = new_head_k * bias - tail_k

        # Aggregate scores using p-norm
        score = torch.stack([score_i, score_j, score_k], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        
        return score


class Rotate4D(KGEModel):
    """
    Rotate4D model: represents relations as rotations in 4D space using quaternions.
    
    Uses dual quaternion rotations in 4D space for more expressive relation modeling.
    """
    
    def __init__(
        self, 
        num_entity: int, 
        num_relation: int, 
        hidden_dim: int, 
        gamma: float, 
        p_norm: int = 2
    ):
        """
        Initialize Rotate4D model.
        
        Args:
            num_entity: Number of entities
            num_relation: Number of relations
            hidden_dim: Dimension of embeddings
            gamma: Margin parameter
            p_norm: P-norm for distance calculation
        """
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm
        self.pi = math.pi

        # Margin parameter (fixed)
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        # Embedding range for initialization
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        # Entity embeddings: quaternion (u, x, y, z)
        self.entity_embedding = nn.Parameter(
            torch.zeros(num_entity, hidden_dim * 4)
        )
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Relation embeddings: dual quaternion rotation parameters + bias
        # (alpha_1, alpha_2, alpha_3, beta_1, beta_2, beta_3, bias)
        self.relation_embedding = nn.Parameter(
            torch.zeros(num_relation, hidden_dim * 7)
        )
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Initialize bias component to 1
        nn.init.ones_(
            tensor=self.relation_embedding[:, 6*hidden_dim:7*hidden_dim]
        )

    def func(
        self, 
        head: torch.Tensor, 
        rel: torch.Tensor, 
        tail: torch.Tensor, 
        batch_type: BatchType
    ) -> torch.Tensor:
        """
        Scoring function using 4D quaternion rotations.
        
        Implements dual quaternion rotation in 4D space.
        """
        # Decompose entity embeddings into quaternion components
        u_h, x_h, y_h, z_h = torch.chunk(head, 4, dim=2)
        u_t, x_t, y_t, z_t = torch.chunk(tail, 4, dim=2)
        
        # Decompose relation embeddings
        alpha_1, alpha_2, alpha_3, beta_1, beta_2, beta_3, bias = torch.chunk(rel, 7, dim=2)
        bias = torch.abs(bias)  # Ensure positive scaling

        # Normalize angles to [-pi, pi]
        alpha_1 = alpha_1 / (self.embedding_range.item() / self.pi)
        alpha_2 = alpha_2 / (self.embedding_range.item() / self.pi)
        alpha_3 = alpha_3 / (self.embedding_range.item() / self.pi)
        beta_1 = beta_1 / (self.embedding_range.item() / self.pi)
        beta_2 = beta_2 / (self.embedding_range.item() / self.pi)
        beta_3 = beta_3 / (self.embedding_range.item() / self.pi)

        # Compute first quaternion rotation parameters
        a_r = torch.cos(alpha_1) * torch.sin(alpha_3)
        b_r = torch.sin(alpha_1) * torch.sin(alpha_3)
        c_r = torch.cos(alpha_2) * torch.cos(alpha_3)
        d_r = torch.sin(alpha_2) * torch.cos(alpha_3)

        # Compute second quaternion rotation parameters
        p_r = torch.cos(beta_1) * torch.sin(beta_3)
        q_r = torch.sin(beta_1) * torch.sin(beta_3)
        r_r = torch.cos(beta_2) * torch.cos(beta_3)
        s_r = torch.sin(beta_2) * torch.cos(beta_3)

        # First quaternion multiplication: q1 * h
        A = a_r * u_h - b_r * x_h - c_r * y_h - d_r * z_h
        B = a_r * x_h + b_r * u_h + c_r * z_h - d_r * y_h
        C = a_r * y_h - b_r * z_h + c_r * u_h + d_r * x_h
        D = a_r * z_h + b_r * y_h - c_r * x_h + d_r * u_h

        # Second quaternion multiplication: (q1 * h) * q2
        new_u_h = A * p_r - B * q_r - C * r_r - D * s_r
        new_x_h = A * q_r + B * p_r + C * s_r - D * r_r
        new_y_h = A * r_r - B * s_r + C * p_r + D * q_r
        new_z_h = A * s_r + B * r_r - C * q_r + D * p_r

        # Compute distance scores
        score_u = new_u_h * bias - u_t
        score_x = new_x_h * bias - x_t
        score_y = new_y_h * bias - y_t
        score_z = new_z_h * bias - z_t

        # Aggregate scores using p-norm
        score = torch.stack([score_u, score_x, score_y, score_z], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        
        return score

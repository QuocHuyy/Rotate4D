import os
import torch
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Set
from torch.utils.data import Dataset
from collections import defaultdict


class BatchType(Enum):
    """Batch type for positive and negative sampling."""
    HEAD_BATCH = 0
    TAIL_BATCH = 1
    SINGLE = 2


class ModeType(Enum):
    """Mode type for training, validation and testing."""
    TRAIN = 0
    VALID = 1
    TEST = 2


class DataReader(object):
    """
    Reads knowledge graph data from files.
    
    Expected file structure:
    - entities.dict: entity_name \t entity_id
    - relations.dict: relation_name \t relation_id
    - train.txt: head \t relation \t tail
    - valid.txt: head \t relation \t tail
    - test.txt: head \t relation \t tail
    """
    
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Define file paths
        entity_dict_path = os.path.join(data_path, 'entities.dict')
        relation_dict_path = os.path.join(data_path, 'relations.dict')
        train_data_path = os.path.join(data_path, 'train.txt')
        valid_data_path = os.path.join(data_path, 'valid.txt')
        test_data_path = os.path.join(data_path, 'test.txt')
        
        # Validate all required files exist
        for path in [entity_dict_path, relation_dict_path, train_data_path, 
                     valid_data_path, test_data_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Read dictionaries
        self.entity_dict = self.read_dict(entity_dict_path)
        self.relation_dict = self.read_dict(relation_dict_path)
        
        print(f"Loaded {len(self.entity_dict)} entities and {len(self.relation_dict)} relations")
        
        # Read data splits
        self.train_data = self.read_data(train_data_path, self.entity_dict, self.relation_dict)
        self.valid_data = self.read_data(valid_data_path, self.entity_dict, self.relation_dict)
        self.test_data = self.read_data(test_data_path, self.entity_dict, self.relation_dict)
        
        print(f"Loaded {len(self.train_data)} training triples")
        print(f"Loaded {len(self.valid_data)} validation triples")
        print(f"Loaded {len(self.test_data)} test triples")

    def read_dict(self, dict_path: str) -> Dict[str, int]:
        """
        Read entity or relation dictionary.
        
        Args:
            dict_path: Path to the dictionary file
            
        Returns:
            Dictionary mapping element name to ID
        """
        element_dict = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split('\t')
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid format at line {line_num} in {dict_path}: "
                        f"expected 2 tab-separated values, got {len(parts)}"
                    )
                
                id_str, element = parts
                try:
                    element_id = int(id_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid ID at line {line_num} in {dict_path}: "
                        f"'{id_str}' is not a valid integer"
                    )
                
                if element in element_dict:
                    raise ValueError(
                        f"Duplicate element '{element}' found in {dict_path}"
                    )
                
                element_dict[element] = element_id

        return element_dict

    def read_data(
        self, 
        data_path: str, 
        entity_dict: Dict[str, int], 
        relation_dict: Dict[str, int]
    ) -> List[Tuple[int, int, int]]:
        """
        Read train/valid/test data.
        
        Args:
            data_path: Path to the data file
            entity_dict: Entity name to ID mapping
            relation_dict: Relation name to ID mapping
            
        Returns:
            List of triples (head_id, relation_id, tail_id)
        """
        triples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                parts = line.split('\t')
                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid format at line {line_num} in {data_path}: "
                        f"expected 3 tab-separated values, got {len(parts)}"
                    )
                
                head, relation, tail = parts
                
                # Validate entities and relations exist in dictionaries
                if head not in entity_dict:
                    raise ValueError(
                        f"Unknown entity '{head}' at line {line_num} in {data_path}"
                    )
                if tail not in entity_dict:
                    raise ValueError(
                        f"Unknown entity '{tail}' at line {line_num} in {data_path}"
                    )
                if relation not in relation_dict:
                    raise ValueError(
                        f"Unknown relation '{relation}' at line {line_num} in {data_path}"
                    )
                
                triples.append((
                    entity_dict[head],
                    relation_dict[relation],
                    entity_dict[tail]
                ))
                
        return triples


class TrainDataset(Dataset):
    """
    Training dataset with negative sampling.
    
    Supports both head-batch and tail-batch negative sampling strategies.
    """
    
    def __init__(self, data_reader: DataReader, neg_size: int, batch_type: BatchType):
        """
        Initialize training dataset.
        
        Args:
            data_reader: DataReader instance
            neg_size: Number of negative samples per positive sample
            batch_type: BatchType.HEAD_BATCH or BatchType.TAIL_BATCH
        """
        if batch_type not in [BatchType.HEAD_BATCH, BatchType.TAIL_BATCH]:
            raise ValueError(
                f"Invalid batch_type: {batch_type}. "
                f"Must be HEAD_BATCH or TAIL_BATCH for training."
            )
        
        self.triples = data_reader.train_data
        self.len = len(self.triples)
        self.num_entity = len(data_reader.entity_dict)
        self.num_relation = len(data_reader.relation_dict)
        self.neg_size = neg_size
        self.batch_type = batch_type

        # Build mappings for negative sampling
        self.hr_map, self.tr_map, self.hr_freq, self.tr_freq = self._build_maps()

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        """
        Get a training sample.
        
        Returns:
            pos_triple: Positive triple (head, relation, tail)
            neg_triples: Negative samples
            subsampling_weight: Weight for subsampling
            batch_type: Type of batch (head or tail)
        """
        pos_triple = self.triples[idx]
        head, rel, tail = pos_triple

        # Calculate subsampling weight based on frequency
        subsampling_weight = self.hr_freq[(head, rel)] + self.tr_freq[(tail, rel)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        # Generate negative samples
        neg_triples = self._generate_negative_samples(head, rel, tail)

        # Convert to tensors
        pos_triple = torch.LongTensor(pos_triple)
        neg_triples = torch.from_numpy(neg_triples)

        return pos_triple, neg_triples, subsampling_weight, self.batch_type

    def _generate_negative_samples(
        self, 
        head: int, 
        rel: int, 
        tail: int
    ) -> np.ndarray:
        """
        Generate negative samples by corrupting head or tail.
        
        Args:
            head: Head entity ID
            rel: Relation ID
            tail: Tail entity ID
            
        Returns:
            Array of negative entity IDs
        """
        neg_samples = []
        neg_size = 0

        # Keep sampling until we have enough valid negative samples
        while neg_size < self.neg_size:
            # Sample more than needed to account for filtering
            candidates = np.random.randint(
                self.num_entity, 
                size=self.neg_size * 2
            )
            
            # Filter out true triples
            if self.batch_type == BatchType.HEAD_BATCH:
                # When corrupting head, filter out valid heads for (?, rel, tail)
                mask = np.in1d(
                    candidates,
                    self.tr_map[(tail, rel)],
                    assume_unique=True,
                    invert=True
                )
            else:  # TAIL_BATCH
                # When corrupting tail, filter out valid tails for (head, rel, ?)
                mask = np.in1d(
                    candidates,
                    self.hr_map[(head, rel)],
                    assume_unique=True,
                    invert=True
                )

            valid_negs = candidates[mask]
            neg_samples.append(valid_negs)
            neg_size += valid_negs.size

        # Concatenate and trim to exact size
        neg_samples = np.concatenate(neg_samples)[:self.neg_size]
        return neg_samples

    @staticmethod
    def collate_fn(data):
        """Collate function for DataLoader."""
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        batch_type = data[0][3]
        return positive_sample, negative_sample, subsample_weight, batch_type

    def _build_maps(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Build mappings for negative sampling and frequency counting.
        
        Returns:
            hr_map: Dict mapping (head, relation) to set of valid tails
            tr_map: Dict mapping (tail, relation) to set of valid heads
            hr_freq: Dict mapping (head, relation) to frequency
            tr_freq: Dict mapping (tail, relation) to frequency
        """
        hr_map = defaultdict(set)
        tr_map = defaultdict(set)
        hr_freq = defaultdict(lambda: 3)  # Initial count of 3
        tr_freq = defaultdict(lambda: 3)

        for head, rel, tail in self.triples:
            hr_map[(head, rel)].add(tail)
            tr_map[(tail, rel)].add(head)
            hr_freq[(head, rel)] += 1
            tr_freq[(tail, rel)] += 1

        # Convert sets to numpy arrays for efficient lookup
        hr_map = {k: np.array(list(v)) for k, v in hr_map.items()}
        tr_map = {k: np.array(list(v)) for k, v in tr_map.items()}
        
        # Convert defaultdict to regular dict
        hr_freq = dict(hr_freq)
        tr_freq = dict(tr_freq)

        return hr_map, tr_map, hr_freq, tr_freq


class TestDataset(Dataset):
    """
    Test/Validation dataset for link prediction evaluation.
    
    For each triple, generates all possible corruptions and marks
    which ones are true triples for filtered evaluation.
    """
    
    def __init__(
        self, 
        data_reader: DataReader, 
        mode: ModeType, 
        batch_type: BatchType
    ):
        """
        Initialize test dataset.
        
        Args:
            data_reader: DataReader instance
            mode: ModeType.VALID or ModeType.TEST
            batch_type: BatchType.HEAD_BATCH or BatchType.TAIL_BATCH
        """
        if mode not in [ModeType.VALID, ModeType.TEST]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be VALID or TEST."
            )
        
        if batch_type not in [BatchType.HEAD_BATCH, BatchType.TAIL_BATCH]:
            raise ValueError(
                f"Invalid batch_type: {batch_type}. "
                f"Must be HEAD_BATCH or TAIL_BATCH."
            )
        
        # Build set of all true triples for filtering
        self.triple_set = set(
            data_reader.train_data + 
            data_reader.valid_data + 
            data_reader.test_data
        )
        
        # Select appropriate data split
        if mode == ModeType.VALID:
            self.triples = data_reader.valid_data
        else:  # ModeType.TEST
            self.triples = data_reader.test_data

        self.len = len(self.triples)
        self.num_entity = len(data_reader.entity_dict)
        self.num_relation = len(data_reader.relation_dict)
        self.mode = mode
        self.batch_type = batch_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        """
        Get a test sample with all possible corruptions.
        
        Returns:
            positive_sample: The true triple
            negative_sample: All possible entity replacements
            filter_bias: Mask indicating which samples are true triples
            batch_type: Type of batch (head or tail)
        """
        head, relation, tail = self.triples[idx]

        # Generate all possible corruptions
        if self.batch_type == BatchType.HEAD_BATCH:
            # Corrupt head: (?, relation, tail)
            tmp = [
                (0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                else (-1, head) 
                for rand_head in range(self.num_entity)
            ]
            tmp[head] = (0, head)  # Ensure the true head is included
            
        elif self.batch_type == BatchType.TAIL_BATCH:
            # Corrupt tail: (head, relation, ?)
            tmp = [
                (0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                else (-1, tail)
                for rand_tail in range(self.num_entity)
            ]
            tmp[tail] = (0, tail)  # Ensure the true tail is included
            
        else:
            raise ValueError(
                f'Batch type {self.batch_type} not supported for testing'
            )

        # Convert to tensors
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.batch_type

    @staticmethod
    def collate_fn(data):
        """Collate function for DataLoader."""
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    """
    Iterator that alternates between head-batch and tail-batch sampling.
    
    This helps balance the training between predicting heads and tails.
    """
    
    def __init__(self, dataloader_head, dataloader_tail):
        """
        Initialize bidirectional iterator.
        
        Args:
            dataloader_head: DataLoader for head-batch sampling
            dataloader_tail: DataLoader for tail-batch sampling
        """
        self.iterator_head = self._one_shot_iterator(dataloader_head)
        self.iterator_tail = self._one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        """Get next batch, alternating between head and tail."""
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def _one_shot_iterator(dataloader):
        """
        Transform a PyTorch DataLoader into an infinite iterator.
        
        Args:
            dataloader: PyTorch DataLoader
            
        Yields:
            Batches from the dataloader indefinitely
        """
        while True:
            for data in dataloader:
                yield data

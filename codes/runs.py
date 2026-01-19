import os
import sys
import json
import logging
import argparse
from collections import defaultdict
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import KGEModel, Rotate3D, Rotate4D
from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='runs.py [<args>] [-h | --help]'
    )

    # Mode arguments
    parser.add_argument('--do_train', action='store_true', 
                       help='Train the model')
    parser.add_argument('--do_valid', action='store_true',
                       help='Evaluate on validation set')
    parser.add_argument('--do_test', action='store_true',
                       help='Evaluate on test set')
    parser.add_argument('--evaluate_train', action='store_true', 
                       help='Evaluate on training data')

    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to the data directory')
    parser.add_argument('--model', default='Rotate4D', type=str,
                       choices=['Rotate3D', 'Rotate4D'],
                       help='Model name')

    # Model hyperparameters
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int,
                       help='Number of negative samples per positive sample')
    parser.add_argument('-d', '--hidden_dim', default=500, type=int,
                       help='Embedding dimension')
    parser.add_argument('-g', '--gamma', default=12.0, type=float,
                       help='Margin parameter')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float,
                       help='Temperature for adversarial negative sampling')
    parser.add_argument('--use_adversarial_negative_sampling', action='store_true',
                       default=True, help='Use adversarial negative sampling')
    parser.add_argument('--disable_adv', action='store_true', 
                       help='Disable adversarial negative sampling')
    parser.add_argument('-reg', '--regularization', default=0.0, type=float,
                       help='Regularization coefficient')
    parser.add_argument('-p', '--p_norm', default=2, type=int,
                       help='P-norm for computing the distance score')

    # Training arguments
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                       help='Training batch size')
    parser.add_argument('--test_batch_size', default=4, type=int, 
                       help='Validation/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float,
                       help='Initial learning rate')
    parser.add_argument('--max_steps', default=100000, type=int,
                       help='Maximum training steps')
    parser.add_argument('--warm_up_steps', default=None, type=int,
                       help='Warmup steps (default: max_steps // 2)')
    parser.add_argument('--max_grad_norm', default=0.0, type=float,
                       help='Maximum gradient norm for clipping (0 = disabled)')

    # System arguments
    parser.add_argument('-cpu', '--cpu_num', default=4, type=int,
                       help='Number of CPU workers for data loading')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed for reproducibility')

    # Checkpoint arguments
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str,
                       help='Path to checkpoint for initialization/evaluation')
    parser.add_argument('-save', '--save_path', default=None, type=str,
                       help='Path to save model checkpoints')
    parser.add_argument('--save_checkpoint_steps', default=5000, type=int,
                       help='Save checkpoint every N steps')

    # Logging arguments
    parser.add_argument('--valid_steps', default=10000, type=int,
                       help='Validate every N steps during training')
    parser.add_argument('--log_steps', default=100, type=int, 
                       help='Log training metrics every N steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, 
                       help='Log evaluation progress every N steps')

    args = parser.parse_args(args)
    
    # Post-processing: handle disable_adv flag
    if args.disable_adv:
        args.use_adversarial_negative_sampling = False
    
    return args


def validate_args(args):
    """Validate command line arguments."""
    # Check that at least one mode is selected
    if not any([args.do_train, args.do_valid, args.do_test, args.evaluate_train]):
        raise ValueError(
            'At least one of --do_train, --do_valid, --do_test, '
            'or --evaluate_train must be specified'
        )

    # Check data path or checkpoint is provided
    if args.init_checkpoint:
        if not os.path.exists(args.init_checkpoint):
            raise ValueError(f'Checkpoint directory does not exist: {args.init_checkpoint}')
    elif args.data_path is None:
        raise ValueError('Either --data_path or --init_checkpoint must be specified')
    
    if args.data_path and not os.path.exists(args.data_path):
        raise ValueError(f'Data directory does not exist: {args.data_path}')

    # Check save path for training
    if args.do_train and args.save_path is None:
        raise ValueError('--save_path must be specified when --do_train is set')

    # Validate hyperparameters
    if args.hidden_dim <= 0:
        raise ValueError(f'hidden_dim must be positive, got {args.hidden_dim}')
    
    if args.batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {args.batch_size}')
    
    if args.learning_rate <= 0:
        raise ValueError(f'learning_rate must be positive, got {args.learning_rate}')
    
    if args.gamma <= 0:
        raise ValueError(f'gamma must be positive, got {args.gamma}')


def override_config(args):
    """
    Override model and data configuration from checkpoint.
    
    Args:
        args: Argument namespace
    """
    config_path = os.path.join(args.init_checkpoint, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    
    with open(config_path, 'r') as f:
        checkpoint_args = json.load(f)

    # Override essential parameters
    args.model = checkpoint_args['model']
    args.data_path = checkpoint_args['data_path']
    args.hidden_dim = checkpoint_args['hidden_dim']
    args.test_batch_size = checkpoint_args.get('test_batch_size', args.test_batch_size)
    args.p_norm = checkpoint_args.get('p_norm', args.p_norm)
    args.gamma = checkpoint_args.get('gamma', args.gamma)
    
    logging.info('Loaded configuration from checkpoint')


def save_model(model, optimizer, save_variable_list: Dict[str, Any], args):
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        save_variable_list: Dictionary of additional variables to save
        args: Training arguments
    """
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Save configuration
    args_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Save checkpoint
    checkpoint = {
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    checkpoint_path = os.path.join(args.save_path, 'checkpoint')
    torch.save(checkpoint, checkpoint_path)
    
    logging.info(f'Saved checkpoint to {checkpoint_path}')

    # Save embeddings separately for easier analysis
    try:
        entity_embedding = model.module.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'entity_embedding'),
            entity_embedding
        )

        relation_embedding = model.module.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'relation_embedding'),
            relation_embedding
        )
        
        logging.info('Saved embeddings to numpy files')
    except Exception as e:
        logging.warning(f'Failed to save embeddings: {e}')


def set_logger(args):
    """
    Setup logging to both file and console.
    
    Args:
        args: Arguments containing save_path or init_checkpoint
    """
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f'Logging to {log_file}')


def log_metrics(mode: str, step: int, metrics: Dict[str, float]):
    """
    Log evaluation metrics.
    
    Args:
        mode: Mode name (e.g., 'Train', 'Valid', 'Test')
        step: Current training step
        metrics: Dictionary of metric names to values
    """
    logging.info(f'{"="*50}')
    logging.info(f'{mode} Metrics at step {step}')
    logging.info(f'{"="*50}')
    for metric, value in metrics.items():
        logging.info(f'  {metric}: {value:.6f}')
    logging.info(f'{"="*50}')


def log_metrics_per_relation(
    mode: str, 
    step: int, 
    metrics_per_relation: Dict[int, Dict[str, float]], 
    relation_dict: Dict[int, str]
):
    """
    Log per-relation evaluation metrics.
    
    Args:
        mode: Mode name
        step: Current training step
        metrics_per_relation: Dictionary mapping relation ID to metrics
        relation_dict: Dictionary mapping relation ID to relation name
    """
    logging.info(f'{"="*50}')
    logging.info(f'{mode} Per-Relation Metrics at step {step}')
    logging.info(f'{"="*50}')
    
    for rel_id, metrics in metrics_per_relation.items():
        rel_name = relation_dict.get(rel_id, f'Unknown({rel_id})')
        logging.info(f'Relation: {rel_name}')
        for metric, value in metrics.items():
            logging.info(f'  {metric}: {value:.6f}')
        logging.info('-' * 50)


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f'Set random seed to {seed}')


def create_model(args, num_entity: int, num_relation: int) -> KGEModel:
    """
    Create KGE model based on args.
    
    Args:
        args: Model arguments
        num_entity: Number of entities
        num_relation: Number of relations
        
    Returns:
        KGEModel instance
    """
    if args.model == 'Rotate3D':
        model = Rotate3D(
            num_entity, 
            num_relation, 
            args.hidden_dim, 
            args.gamma, 
            args.p_norm
        )
    elif args.model == 'Rotate4D':
        model = Rotate4D(
            num_entity, 
            num_relation, 
            args.hidden_dim, 
            args.gamma, 
            args.p_norm
        )
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    return model


def log_model_info(model: KGEModel, args):
    """Log model architecture and parameters."""
    logging.info(f'{"="*50}')
    logging.info('Model Architecture')
    logging.info(f'{"="*50}')
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        logging.info(
            f'  {name}: {tuple(param.size())}, '
            f'params={num_params:,}, '
            f'requires_grad={param.requires_grad}'
        )
    
    logging.info(f'{"="*50}')
    logging.info(f'Total parameters: {total_params:,}')
    logging.info(f'Trainable parameters: {trainable_params:,}')
    logging.info(f'{"="*50}')


def train(args, kge_model, data_reader, init_step: int = 0):
    """
    Training loop.
    
    Args:
        args: Training arguments
        kge_model: Model to train
        data_reader: DataReader instance
        init_step: Initial step (for resuming training)
    """
    # Create data loaders
    train_dataloader_head = DataLoader(
        TrainDataset(data_reader, args.negative_sample_size, BatchType.HEAD_BATCH),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(data_reader, args.negative_sample_size, BatchType.TAIL_BATCH),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn
    )

    train_iterator = BidirectionalOneShotIterator(
        train_dataloader_head, 
        train_dataloader_tail
    )

    # Setup optimizer
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()),
        lr=current_learning_rate
    )

    # Setup learning rate schedule
    warm_up_steps = args.warm_up_steps if args.warm_up_steps is not None else args.max_steps // 2

    # Load checkpoint if resuming
    if args.init_checkpoint:
        logging.info(f'Loading checkpoint from {args.init_checkpoint}...')
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        init_step = checkpoint.get('step', 0)
        current_learning_rate = checkpoint.get('current_learning_rate', args.learning_rate)
        warm_up_steps = checkpoint.get('warm_up_steps', warm_up_steps)
        
        logging.info(f'Resumed from step {init_step}')

    # Log training configuration
    logging.info(f'{"="*50}')
    logging.info('Training Configuration')
    logging.info(f'{"="*50}')
    logging.info(f'  Initial step: {init_step}')
    logging.info(f'  Max steps: {args.max_steps}')
    logging.info(f'  Batch size: {args.batch_size}')
    logging.info(f'  Learning rate: {current_learning_rate}')
    logging.info(f'  Warm-up steps: {warm_up_steps}')
    logging.info(f'  Negative sample size: {args.negative_sample_size}')
    logging.info(f'  Adversarial sampling: {args.use_adversarial_negative_sampling}')
    logging.info(f'  Adversarial temperature: {args.adversarial_temperature}')
    logging.info(f'  Regularization: {args.regularization}')
    logging.info(f'  Max gradient norm: {args.max_grad_norm}')
    logging.info(f'{"="*50}')

    # Training loop
    training_logs = []
    
    for step in range(init_step, args.max_steps):
        # Training step
        log = kge_model.module.train_step(kge_model, optimizer, train_iterator, args)
        training_logs.append(log)

        # Learning rate decay
        if step == warm_up_steps:
            current_learning_rate = current_learning_rate / 10
            logging.info(f'Reducing learning rate to {current_learning_rate:.2e} at step {step}')
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 3

        # Save checkpoint
        if step % args.save_checkpoint_steps == 0 and step > 0:
            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_model(kge_model, optimizer, save_variable_list, args)

        # Log training metrics
        if step % args.log_steps == 0 and len(training_logs) > 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum(log[metric] for log in training_logs) / len(training_logs)
            
            logging.info(
                f'Step {step}/{args.max_steps}: '
                f'loss={metrics["loss"]:.4f}, '
                f'pos_loss={metrics["positive_sample_loss"]:.4f}, '
                f'neg_loss={metrics["negative_sample_loss"]:.4f}, '
                f'reg={metrics["regularization"]:.4f}'
            )
            training_logs = []

        # Validation
        if args.do_valid and step % args.valid_steps == 0 and step > 0:
            logging.info('Evaluating on validation set...')
            metrics, metrics_per_relation = kge_model.module.test_step(
                kge_model, data_reader, ModeType.VALID, args
            )
            log_metrics('Validation', step, metrics)

        # Test during training (if requested)
        if args.do_test and step % args.valid_steps == 0 and step > 0:
            logging.info('Evaluating on test set...')
            metrics, metrics_per_relation = kge_model.module.test_step(
                kge_model, data_reader, ModeType.TEST, args
            )
            log_metrics('Test', step, metrics)

    # Save final model
    save_variable_list = {
        'step': args.max_steps,
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    save_model(kge_model, optimizer, save_variable_list, args)
    logging.info('Training completed!')


def main(args):
    """Main execution function."""
    # Validate arguments
    validate_args(args)

    # Override config if loading from checkpoint
    if args.init_checkpoint:
        override_config(args)

    # Create save directory
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    # Setup logging
    set_logger(args)

    # Set random seed
    set_random_seed(args.seed)

    # Load data
    logging.info(f'Loading data from {args.data_path}...')
    data_reader = DataReader(args.data_path)
    
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)
    
    # Create reverse mapping for relation names
    relation_dict = {v: k for k, v in data_reader.relation_dict.items()}

    # Log data statistics
    logging.info(f'{"="*50}')
    logging.info('Data Statistics')
    logging.info(f'{"="*50}')
    logging.info(f'  Model: {args.model}')
    logging.info(f'  Data path: {args.data_path}')
    logging.info(f'  Entities: {num_entity:,}')
    logging.info(f'  Relations: {num_relation:,}')
    logging.info(f'  Training triples: {len(data_reader.train_data):,}')
    logging.info(f'  Validation triples: {len(data_reader.valid_data):,}')
    logging.info(f'  Test triples: {len(data_reader.test_data):,}')
    logging.info(f'{"="*50}')

    # Create model
    kge_model = create_model(args, num_entity, num_relation)
    
    # Log model info
    log_model_info(kge_model, args)

    # Move to GPU if available
    if args.cuda and torch.cuda.is_available():
        kge_model = torch.nn.DataParallel(kge_model)
        kge_model = kge_model.cuda()
        logging.info(f'Using {torch.cuda.device_count()} GPUs')
    else:
        logging.warning('CUDA not available, using CPU')

    # Training
    if args.do_train:
        train(args, kge_model, data_reader)

    # Load checkpoint for evaluation if not training
    if not args.do_train and args.init_checkpoint:
        logging.info(f'Loading checkpoint from {args.init_checkpoint}...')
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 0)
    else:
        step = args.max_steps if args.do_train else 0

    # Validation
    if args.do_valid:
        logging.info('Evaluating on validation set...')
        metrics, metrics_per_relation = kge_model.module.test_step(
            kge_model, data_reader, ModeType.VALID, args
        )
        log_metrics('Validation', step, metrics)

    # Testing
    if args.do_test:
        logging.info('Evaluating on test set...')
        metrics, metrics_per_relation = kge_model.module.test_step(
            kge_model, data_reader, ModeType.TEST, args
        )
        log_metrics('Test', step, metrics)
        log_metrics_per_relation('Test', step, metrics_per_relation, relation_dict)

    # Evaluate on training data
    if args.evaluate_train:
        logging.info('Evaluating on training set...')
        metrics, metrics_per_relation = kge_model.module.test_step(
            kge_model, data_reader, ModeType.TRAIN, args
        )
        log_metrics('Training', step, metrics)
        log_metrics_per_relation('Training', step, metrics_per_relation, relation_dict)

    logging.info('All tasks completed successfully!')


if __name__ == '__main__':
    try:
        main(parse_args())
    except Exception as e:
        logging.error(f'Fatal error: {e}', exc_info=True)
        sys.exit(1)

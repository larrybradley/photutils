# PSFPhotometry Parallelization

## Overview

The `PSFPhotometry` class has been optimized with parallel processing capabilities to improve performance when fitting multiple sources or source groups.

## New Parameter: `n_jobs`

A new parameter `n_jobs` has been added to both `PSFPhotometry` and `IterativePSFPhotometry`:

- **Default**: `1` (no parallelization)
- **`None`**: Uses all available CPU cores
- **`> 1`**: Uses the specified number of worker threads
- **`< 1`**: Raises `ValueError`

## Implementation Details

### Parallel Processing Strategy

1. **ProcessPoolExecutor**: Uses `ProcessPoolExecutor` for true parallel processing:
   - Bypasses Python's GIL (Global Interpreter Lock) for CPU-intensive fitting
   - Provides true parallel execution for numerical computations
   - Includes model serialization to handle Astropy model objects
   - Serializes model parameters and reconstructs models in worker processes

2. **Group-level Parallelization**: Each source group is processed in parallel:
   - Individual sources within a group are still processed together (as required)
   - Different groups can be processed simultaneously
   - Maintains the same grouping logic and results

3. **Fallback to Sequential**: When `n_jobs=1`, the original sequential implementation is used.

### Performance Benefits

- **True parallel processing**: Bypasses Python's GIL for CPU-intensive fitting operations
- **Ideal for many sources**: Best performance improvement when processing many independent sources
- **Grouped sources**: Still benefits from parallelization when multiple groups exist
- **CPU-bound operations**: Fitting operations can utilize multiple CPU cores simultaneously

### Usage Examples

```python
# Sequential processing (default)
psf = PSFPhotometry(psf_model, fit_shape, finder=finder, n_jobs=1)

# Parallel processing with 4 threads
psf = PSFPhotometry(psf_model, fit_shape, finder=finder, n_jobs=4)

# Use all available cores
psf = PSFPhotometry(psf_model, fit_shape, finder=finder, n_jobs=None)

# IterativePSFPhotometry also supports parallelization
iter_psf = IterativePSFPhotometry(psf_model, fit_shape, finder=finder, n_jobs=4)
```

## Backward Compatibility

- The `n_jobs` parameter defaults to `1`, maintaining backward compatibility
- All existing functionality is preserved
- Results are identical between sequential and parallel processing

## Technical Implementation

### Model Serialization
- `_serialize_model_params()`: Extracts model class, parameters, and constraints
- `_deserialize_model()`: Reconstructs model from serialized information
- `_worker_fit_single_group()`: Static method for multiprocessing compatibility

### Process Management
- Uses `ProcessPoolExecutor` for true parallel execution
- Each worker process receives serialized model parameters
- Results are collected and ordered to maintain consistency

## Error Handling

- Exceptions during parallel processing are properly caught and re-raised with context
- The implementation maintains the same error behavior as the sequential version

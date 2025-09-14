#!/usr/bin/env python
"""
Benchmark script to test parallel vs sequential EPSFFitter performance.

This script tests different numbers of stars and jobs to determine when
parallel processing becomes beneficial over sequential processing.
"""
import time
import warnings

import numpy as np
from astropy.nddata import NDData
from astropy.table import Table

from photutils.datasets import make_100gaussians_image
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder, EPSFFitter, extract_stars


def create_test_data(n_stars=50):
    """
    Create test data with known stars for benchmarking.
    """
    print(f"Creating test data with {n_stars} target stars...")

    # Create synthetic image
    data = make_100gaussians_image()
    nddata = NDData(data=data)

    # Find stars using DAOStarFinder
    finder = DAOStarFinder(fwhm=3.0, threshold=5.0)
    sources = finder(data)

    # Select well-positioned stars (away from edges)
    # Filter sources to avoid edge effects
    margin = 20
    h, w = data.shape
    mask = ((sources['xcentroid'] > margin) &
            (sources['xcentroid'] < w - margin) &
            (sources['ycentroid'] > margin) &
            (sources['ycentroid'] < h - margin))

    filtered_sources = sources[mask]

    # Take the requested number of brightest stars
    if len(filtered_sources) > n_stars:
        # Sort by peak intensity and take the brightest
        filtered_sources.sort('peak')
        filtered_sources.reverse()
        filtered_sources = filtered_sources[:n_stars]

    # Extract star cutouts
    stars_tbl = Table()
    stars_tbl['x'] = filtered_sources['xcentroid']
    stars_tbl['y'] = filtered_sources['ycentroid']

    # Use slightly smaller cutouts to avoid edge issues
    stars = extract_stars(nddata, stars_tbl, size=21)

    print(f"Successfully extracted {len(stars)} stars")
    return data, stars


def build_initial_epsf(stars):
    """
    Build an initial ePSF for fitting tests.
    """
    print('Building initial ePSF...')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        epsf_builder = EPSFBuilder(oversampling=1, maxiters=8,
                                   progress_bar=False,
                                   recentering_maxiters=3)
        try:
            epsf, fitted_stars = epsf_builder(stars)
            print(f"ePSF built successfully with {len(fitted_stars)} stars")
            return epsf
        except ValueError as e:
            print(f"ePSF building failed: {e}")
            return None


def benchmark_fitting(epsf, stars, n_jobs_list=[1, 2, 4, 8], n_runs=3):
    """
    Benchmark EPSFFitter with different numbers of jobs.
    """
    print(f"\nBenchmarking with {len(stars)} stars...")
    print(f"Jobs to test: {n_jobs_list}")
    print(f"Runs per configuration: {n_runs}")
    print('-' * 60)

    results = {}

    for n_jobs in n_jobs_list:
        print(f"Testing n_jobs={n_jobs}...")
        times = []

        for run in range(n_runs):
            fitter = EPSFFitter(n_jobs=n_jobs)

            # Suppress warnings during timing
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                start_time = time.time()
                try:
                    fitted_stars = fitter(epsf, stars)
                    end_time = time.time()

                    # Verify we got results
                    if len(fitted_stars) == len(stars):
                        times.append(end_time - start_time)
                        print(f"  Run {run + 1}: {times[-1]:.3f}s")
                    else:
                        print(f"  Run {run + 1}: FAILED (got {len(fitted_stars)}/{len(stars)} stars)")

                except Exception as e:
                    print(f"  Run {run + 1}: ERROR - {e}")

        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[n_jobs] = {
                'times': times,
                'mean': avg_time,
                'std': std_time,
                'success_rate': len(times) / n_runs,
            }
            print(f"  Average: {avg_time:.3f} ± {std_time:.3f}s "
                  f"(success: {len(times)}/{n_runs})")
        else:
            print(f"  No successful runs for n_jobs={n_jobs}")

        print()

    return results


def analyze_results(results):
    """
    Analyze and display benchmark results.
    """
    print('=' * 60)
    print('BENCHMARK RESULTS SUMMARY')
    print('=' * 60)

    if not results:
        print('No successful benchmark results to analyze.')
        return

    # Find baseline (n_jobs=1) for comparison
    baseline = None
    for n_jobs in sorted(results.keys()):
        if n_jobs == 1:
            baseline = results[n_jobs]['mean']
            break

    print(f"{'Jobs':<6} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Success':<10}")
    print('-' * 60)

    for n_jobs in sorted(results.keys()):
        result = results[n_jobs]
        time_str = f"{result['mean']:.3f}±{result['std']:.3f}"

        if baseline and baseline > 0:
            speedup = baseline / result['mean']
            efficiency = speedup / n_jobs * 100
            speedup_str = f"{speedup:.2f}x"
            efficiency_str = f"{efficiency:.1f}%"
        else:
            speedup_str = 'N/A'
            efficiency_str = 'N/A'

        success_str = f"{result['success_rate']*100:.0f}%"

        print(f"{n_jobs:<6} {time_str:<12} {speedup_str:<10} {efficiency_str:<12} {success_str:<10}")

    # Recommendations
    print('\nRECOMMENDations:')
    if baseline:
        best_speedup = 1.0
        best_n_jobs = 1

        for n_jobs, result in results.items():
            if n_jobs > 1 and result['success_rate'] > 0.8:  # At least 80% success
                speedup = baseline / result['mean']
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_n_jobs = n_jobs

        if best_speedup > 1.1:  # At least 10% improvement
            print(f"• Best configuration: n_jobs={best_n_jobs} "
                  f"({best_speedup:.2f}x speedup)")
        else:
            print('• Sequential processing (n_jobs=1) is recommended')
            print('  Parallel processing overhead exceeds benefits for this dataset')

        # Efficiency analysis
        efficient_configs = []
        for n_jobs, result in results.items():
            if n_jobs > 1 and result['success_rate'] > 0.8:
                speedup = baseline / result['mean']
                efficiency = speedup / n_jobs
                if efficiency > 0.7:  # >70% efficiency
                    efficient_configs.append((n_jobs, efficiency))

        if efficient_configs:
            efficient_configs.sort(key=lambda x: x[1], reverse=True)
            best_eff_n_jobs, best_eff = efficient_configs[0]
            print(f"• Most efficient: n_jobs={best_eff_n_jobs} "
                  f"({best_eff*100:.1f}% efficiency)")

    print("\nNote: Parallel processing benefits depend on:")
    print("• Number of stars (more stars = better parallelization)")
    print("• System CPU cores and memory")
    print("• Process startup overhead vs computation time")


def main():
    """
    Main benchmark function.
    """
    print('EPSFFitter Parallel Processing Benchmark')
    print('=' * 60)

    # Test different dataset sizes
    star_counts = [10, 25, 50, 100]

    for n_stars in star_counts:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {n_stars} STARS")
        print(f"{'='*60}")

        # Create test data
        data, stars = create_test_data(n_stars)

        if len(stars) < 5:
            print(f"Insufficient stars extracted ({len(stars)}), skipping...")
            continue

        # Build initial ePSF
        epsf = build_initial_epsf(stars)
        if epsf is None:
            print('Failed to build ePSF, skipping...')
            continue

        # Determine job counts to test based on available CPUs
        import os
        max_cpus = min(os.cpu_count(), 8)  # Limit to 8 for reasonable test time
        n_jobs_list = [1, 2, 4]
        if max_cpus >= 8:
            n_jobs_list.append(8)

        # Benchmark fitting
        results = benchmark_fitting(epsf, stars, n_jobs_list, n_runs=3)

        # Analyze results
        analyze_results(results)

    print(f"\n{'='*60}")
    print('BENCHMARK COMPLETE')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

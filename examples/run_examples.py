#!/usr/bin/env python3
"""
Graphizy Examples Runner

This script helps you run all examples or specific ones with proper setup.

Usage:
  python run_examples.py                    # Run all examples
  python run_examples.py 1                  # Run basic usage only
  python run_examples.py 1 2               # Run basic usage and graph metrics
  python run_examples.py --quick           # Run quick versions of all examples
  python run_examples.py --list            # List available examples
"""

import sys
import subprocess
import argparse
from pathlib import Path

def get_examples():
    """Get list of available examples"""
    examples_dir = Path(__file__).parent
    examples = [
        {
            'id': 1,
            'file': '1_basic_usage.py',
            'name': 'Basic Usage',
            'description': 'Fundamental graph operations and visualizations',
            'duration': '~2 minutes'
        },
        {
            'id': 2, 
            'file': '2_graph_metrics.py',
            'name': 'Graph Metrics',
            'description': 'Comprehensive graph analysis and metrics computation',
            'duration': '~3 minutes'
        },
        {
            'id': 3,
            'file': '3_advanced_memory.py', 
            'name': 'Advanced Memory',
            'description': 'Memory functionality and temporal graph analysis',
            'duration': '~4 minutes'
        },
        {
            'id': 4,
            'file': '4_interactive_demo.py',
            'name': 'Interactive video analysis',
            'description': 'Complete simulation with dynamic networks and movie generation',
            'duration': '~5-10 minutes',
        },
        {
            'id': 5,
            'file': '5_add_new_graph_type.py',
            'name': 'Add custom graph',
            'description': 'Example of custom graph addition',
            'duration': '~5-10 minutes',
        },
        {
            'id': 6,
            'file': '6_stream_example.py',
            'name': 'Stream example',
            'description': 'Example of stream',
            'duration': '~5-10 minutes',
        }
    ]
    return examples

def list_examples():
    """List all available examples"""
    examples = get_examples()
    
    print("📚 Available Graphizy Examples:")
    print("=" * 50)
    
    for ex in examples:
        print(f"\n{ex['id']}. {ex['name']} ({ex['duration']})")
        print(f"   File: {ex['file']}")
        print(f"   {ex['description']}")
    
    print(f"\nUsage:")
    print(f"  python run_examples.py 1 2    # Run examples 1 and 2")
    print(f"  python run_examples.py --quick # Quick versions")

def run_example(example, quick=False):
    """Run a specific example"""
    examples_dir = Path(__file__).parent
    script_path = examples_dir / example['file']
    
    if not script_path.exists():
        print(f"❌ Example file not found: {script_path}")
        return False
    
    print(f"\n🚀 Running: {example['name']}")
    print(f"📁 File: {example['file']}")
    print("-" * 40)
    
    try:
        # Prepare command
        cmd = [sys.executable, str(script_path)]
        
        # Add quick args if available and requested
        if quick and 'quick_args' in example:
            cmd.extend(example['quick_args'])
            print(f"⚡ Quick mode: {' '.join(example['quick_args'])}")
        
        # Run the example
        result = subprocess.run(cmd, cwd=examples_dir)
        
        if result.returncode == 0:
            print(f"✅ {example['name']} completed successfully!")
            return True
        else:
            print(f"❌ {example['name']} failed with exit code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⏹️ {example['name']} interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running {example['name']}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run Graphizy examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py              # Run all examples
  python run_examples.py 1 3          # Run examples 1 and 3
  python run_examples.py --quick      # Quick versions of all examples
  python run_examples.py --list       # List available examples
        """
    )
    
    parser.add_argument('examples', nargs='*', type=int,
                       help='Example numbers to run (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available examples')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick versions (where available)')
    
    args = parser.parse_args()
    
    examples = get_examples()
    
    if args.list:
        list_examples()
        return 0
    
    # Determine which examples to run
    if args.examples:
        # Run specific examples
        examples_to_run = []
        for ex_id in args.examples:
            example = next((ex for ex in examples if ex['id'] == ex_id), None)
            if example:
                examples_to_run.append(example)
            else:
                print(f"❌ Invalid example number: {ex_id}")
                return 1
    else:
        # Run all examples
        examples_to_run = examples
    
    if not examples_to_run:
        print("❌ No valid examples to run")
        return 1
    
    # Welcome message
    print("🎨 Graphizy Examples Runner")
    print("=" * 40)
    
    if args.quick:
        print("⚡ Running in quick mode")
    
    print(f"📋 Will run {len(examples_to_run)} example(s):")
    for ex in examples_to_run:
        duration = ex['duration']
        if args.quick and 'quick_args' in ex:
            duration = "~1-2 minutes (quick)"
        print(f"   • {ex['name']} ({duration})")
    
    # Run examples
    successful = 0
    total = len(examples_to_run)
    
    for i, example in enumerate(examples_to_run, 1):
        print(f"\n{'='*50}")
        print(f"📖 Example {i}/{total}: {example['name']}")
        print(f"{'='*50}")
        
        if run_example(example, quick=args.quick):
            successful += 1
        else:
            print(f"\n❌ Example {example['name']} failed!")
            
            # Ask user if they want to continue
            if i < total:
                try:
                    response = input(f"\nContinue with remaining examples? (y/n): ").strip().lower()
                    if response not in ['y', 'yes']:
                        break
                except KeyboardInterrupt:
                    break
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful == total:
        print(f"\n🎉 All examples completed successfully!")
        print(f"📁 Check the examples/output/ directory for generated files")
    elif successful > 0:
        print(f"\n⚠️ Some examples completed successfully")
    else:
        print(f"\n💥 All examples failed")
    
    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())

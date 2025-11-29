#!/usr/bin/env python3
"""
Bundled vLLM + Evaluation Script for Reverse Binary Queries

This script automates the entire workflow:
1. Starts vLLM server (if not already running)
2. Waits for server to be ready
3. Runs evaluation on reverse_queries_binary.csv
4. Optionally shuts down server when done

Usage:
    # Run with auto-start Qwen server
    python run_reverse_binary_bundled.py --model qwen --architecture format_a_qwen --test-size 100

    # Run with LLAMA3 and keep server running
    python run_reverse_binary_bundled.py --model llama3 --architecture enhanced_graphrag_llama3 --keep-server

    # Run all architectures with auto-cleanup
    python run_reverse_binary_bundled.py --model qwen --architecture all --auto-shutdown
"""

import argparse
import json
import logging
import subprocess
import time
import signal
import sys
import os
import requests
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_reverse_binary import ReverseBinaryEvaluator

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLLMServerManager:
    """Manages vLLM server lifecycle (start, health check, shutdown)"""

    # Server configurations
    SERVERS = {
        'qwen': {
            'model': 'Qwen/Qwen2.5-7B-Instruct',
            'port': 8002,
            'gpus': '0,1,2,3',
            'tensor_parallel': 4,
            'venv_python': '/home/omeerdogan23/drugRAG/.venv/bin/python'
        },
        'llama3': {
            'model': 'meta-llama/Llama-3.1-8B-Instruct',
            'port': 8003,
            'gpus': '0,1,2,3',
            'tensor_parallel': 4,
            'venv_python': '/home/omeerdogan23/drugRAG/.venv/bin/python'
        }
    }

    def __init__(self, model_name: str):
        """Initialize server manager for specific model"""
        if model_name not in self.SERVERS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.SERVERS.keys())}")

        self.model_name = model_name
        self.config = self.SERVERS[model_name]
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{self.config['port']}"

    def is_server_running(self) -> bool:
        """Check if server is already running and healthy"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                models = response.json()
                # Check if our model is listed
                if 'data' in models:
                    for model in models['data']:
                        if self.config['model'] in model.get('id', ''):
                            logger.info(f"✅ {self.model_name.upper()} server already running on port {self.config['port']}")
                            return True
                return False
            return False
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
            return False

    def cleanup_stuck_processes(self):
        """Kill any stuck processes on the port"""
        port = self.config['port']
        logger.info(f"🧹 Cleaning up any stuck processes on port {port}...")
        try:
            subprocess.run(
                ['pkill', '-9', '-f', f'port {port}'],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def start_server(self, background: bool = True, timeout: int = 600) -> bool:
        """Start vLLM server"""
        if self.is_server_running():
            logger.info("Server already running, skipping startup")
            return True

        logger.info(f"🚀 Starting {self.model_name.upper()} vLLM server...")
        logger.info(f"   Model: {self.config['model']}")
        logger.info(f"   Port: {self.config['port']}")
        logger.info(f"   GPUs: {self.config['gpus']} (tensor-parallel-size={self.config['tensor_parallel']})")
        logger.info(f"   Timeout: {timeout}s")

        # Cleanup first
        self.cleanup_stuck_processes()

        # Build command
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.config['gpus']

        cmd = [
            self.config['venv_python'],
            '-m', 'vllm.entrypoints.openai.api_server',
            '--model', self.config['model'],
            '--tensor-parallel-size', str(self.config['tensor_parallel']),
            '--host', '0.0.0.0',
            '--port', str(self.config['port']),
            '--dtype', 'float16',
            '--max-model-len', '4096',
            '--gpu-memory-utilization', '0.90',
            '--enable-chunked-prefill',
            '--max-num-batched-tokens', '8192',
            '--enforce-eager',
            '--max-num-seqs', '256',
            '--distributed-executor-backend', 'mp'
        ]

        try:
            # Start process
            if background:
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True  # Detach from parent process
                )
                logger.info(f"   Process started with PID: {self.process.pid}")
            else:
                # Blocking mode for debugging
                subprocess.run(cmd, env=env)

            # Wait for server to be ready
            return self.wait_for_ready(timeout=timeout)

        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False

    def wait_for_ready(self, timeout: int = 600) -> bool:
        """Wait for server to be ready"""
        logger.info(f"⏳ Waiting for server to be ready (timeout: {timeout}s)...")

        start_time = time.time()
        last_log_time = start_time

        while time.time() - start_time < timeout:
            try:
                if self.is_server_running():
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Server ready in {elapsed:.1f}s")
                    return True
            except Exception:
                pass

            # Log progress every 10 seconds
            if time.time() - last_log_time >= 10:
                elapsed = time.time() - start_time
                logger.info(f"   Still waiting... ({elapsed:.0f}s / {timeout}s)")
                last_log_time = time.time()

            time.sleep(2)

        logger.error(f"❌ Server failed to start within {timeout}s")
        return False

    def shutdown_server(self):
        """Gracefully shutdown the server"""
        if self.process is None:
            logger.info("No server process to shutdown")
            return

        logger.info(f"🛑 Shutting down {self.model_name.upper()} server...")

        try:
            # Try graceful shutdown first
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info("   Server shutdown gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("   Graceful shutdown timed out, forcing kill...")
                self.process.kill()
                self.process.wait()
                logger.info("   Server killed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        self.process = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto shutdown"""
        self.shutdown_server()


def main():
    parser = argparse.ArgumentParser(
        description="Bundled vLLM + Evaluation for Reverse Binary Queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with auto-start and auto-shutdown
  python run_reverse_binary_bundled.py --model qwen --architecture format_a_qwen --test-size 100 --auto-shutdown

  # Full evaluation, keep server running for reuse
  python run_reverse_binary_bundled.py --model llama3 --architecture enhanced_graphrag_llama3 --keep-server

  # Run all architectures with one model
  python run_reverse_binary_bundled.py --model qwen --architecture all --test-size 500
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['qwen', 'llama3'],
        help='LLM model to use (qwen or llama3)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        help='Architecture to evaluate (e.g., format_a_qwen, enhanced_graphrag_llama3, all)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help='Number of queries to test (omit for all 1,200)'
    )
    parser.add_argument(
        '--keep-server',
        action='store_true',
        help='Keep server running after evaluation (useful for multiple runs)'
    )
    parser.add_argument(
        '--auto-shutdown',
        action='store_true',
        help='Automatically shutdown server after evaluation'
    )
    parser.add_argument(
        '--server-timeout',
        type=int,
        default=600,
        help='Timeout in seconds for server startup (default: 600)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for results'
    )

    args = parser.parse_args()

    # Banner
    print("=" * 80)
    print("  BUNDLED VLLM + EVALUATION FOR REVERSE BINARY QUERIES")
    print("=" * 80)
    print(f"Model:        {args.model.upper()}")
    print(f"Architecture: {args.architecture}")
    print(f"Test Size:    {args.test_size if args.test_size else 'All 1,200 queries'}")
    print(f"Auto-shutdown: {args.auto_shutdown}")
    print("=" * 80)
    print()

    # Determine if we should shutdown
    should_shutdown = args.auto_shutdown and not args.keep_server

    try:
        # Start vLLM server
        with VLLMServerManager(args.model) as server:
            # Start server (or verify it's running)
            if not server.start_server(timeout=args.server_timeout):
                logger.error("Failed to start server, aborting")
                return 1

            # Run evaluation
            logger.info("\n" + "=" * 80)
            logger.info("STARTING EVALUATION")
            logger.info("=" * 80 + "\n")

            evaluator = ReverseBinaryEvaluator()
            result = evaluator.evaluate(args.architecture, args.test_size)

            # Save results
            output_file = args.output or f"results_reverse_binary_{args.architecture}.json"
            evaluator.save_results(result, output_file)

            # Keep server running if requested
            if args.keep_server:
                logger.info("\n" + "=" * 80)
                logger.info("✅ EVALUATION COMPLETE - Server still running")
                logger.info(f"   Server URL: {server.base_url}")
                logger.info(f"   To shutdown manually, run: pkill -9 -f 'port {server.config['port']}'")
                logger.info("=" * 80)
                # Prevent auto-shutdown by clearing the process reference
                server.process = None
            elif should_shutdown:
                # Auto-shutdown will happen via context manager __exit__
                logger.info("\n🛑 Auto-shutdown enabled, cleaning up...")

        logger.info("\n✅ All done!")
        return 0

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

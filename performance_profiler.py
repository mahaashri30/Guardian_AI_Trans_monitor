"""
Performance Profiling Script for GuardianAI ML Service
Measures latency, throughput, and resource usage
"""

import time
import statistics
import json
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import psutil
import os

class PerformanceProfiler:
    """Profile GuardianAI ML Service performance"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results = {
            'latency_tests': [],
            'throughput_tests': [],
            'resource_usage': []
        }
    
    def generate_test_transaction(self, scenario: str = "normal"):
        """Generate test transaction data"""
        base_time = datetime.now()
        
        if scenario == "normal":
            return {
                "user_id": f"perf_user_{int(time.time())}",
                "amount": 5000,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": base_time.isoformat() + "Z",
                "transaction_id": f"perf_txn_{int(time.time() * 1000)}",
                "user_transaction_history": [
                    {
                        "amount": 4500,
                        "merchant_id": "supplier_a",
                        "device_id": "phone_001",
                        "ip_country": "IN",
                        "timestamp": (base_time - timedelta(hours=1)).isoformat() + "Z"
                    }
                ]
            }
        elif scenario == "suspicious":
            return {
                "user_id": f"perf_user_{int(time.time())}",
                "amount": 75000,
                "merchant_id": "unknown_merchant",
                "device_id": "new_device",
                "ip_country": "XX",
                "timestamp": "2025-12-27T02:00:00Z",
                "transaction_id": f"perf_txn_{int(time.time() * 1000)}",
                "user_transaction_history": [
                    {
                        "amount": 2000,
                        "merchant_id": "supplier_a",
                        "device_id": "phone_001",
                        "ip_country": "IN",
                        "timestamp": (base_time - timedelta(days=1)).isoformat() + "Z"
                    }
                ]
            }
    
    def test_single_request_latency(self, num_requests: int = 100):
        """Test single request latency"""
        print(f"Testing single request latency ({num_requests} requests)...")
        
        latencies = []
        
        for i in range(num_requests):
            transaction = self.generate_test_transaction()
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/fraud-check",
                    json=transaction,
                    timeout=5
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    
                    # Also get reported latency from response
                    result = response.json()
                    reported_latency = result.get('latency_ms', 0)
                    
                    if i % 10 == 0:
                        print(f"Request {i+1}: {latency_ms:.1f}ms (reported: {reported_latency:.1f}ms)")
                else:
                    print(f"Request {i+1} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"Request {i+1} error: {e}")
        
        if latencies:
            results = {
                'num_requests': len(latencies),
                'mean_latency_ms': statistics.mean(latencies),
                'median_latency_ms': statistics.median(latencies),
                'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
                'p99_latency_ms': sorted(latencies)[int(len(latencies) * 0.99)],
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'under_200ms': sum(1 for l in latencies if l < 200) / len(latencies) * 100
            }
            
            self.results['latency_tests'].append(results)
            
            print(f"\nLatency Results:")
            print(f"Mean: {results['mean_latency_ms']:.1f}ms")
            print(f"Median: {results['median_latency_ms']:.1f}ms")
            print(f"P95: {results['p95_latency_ms']:.1f}ms")
            print(f"P99: {results['p99_latency_ms']:.1f}ms")
            print(f"Under 200ms: {results['under_200ms']:.1f}%")
            
            return results
        
        return None
    
    def test_concurrent_throughput(self, num_threads: int = 10, requests_per_thread: int = 50):
        """Test concurrent request throughput"""
        print(f"Testing concurrent throughput ({num_threads} threads, {requests_per_thread} requests each)...")
        
        def worker_thread(thread_id: int):
            """Worker thread for concurrent requests"""
            thread_latencies = []
            
            for i in range(requests_per_thread):
                transaction = self.generate_test_transaction()
                
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}/fraud-check",
                        json=transaction,
                        timeout=10
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        latency_ms = (end_time - start_time) * 1000
                        thread_latencies.append(latency_ms)
                        
                except Exception as e:
                    print(f"Thread {thread_id} request {i+1} error: {e}")
            
            return thread_latencies
        
        # Start concurrent requests
        start_time = time.time()
        all_latencies = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                thread_latencies = future.result()
                all_latencies.extend(thread_latencies)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if all_latencies:
            successful_requests = len(all_latencies)
            throughput_rps = successful_requests / total_time
            
            results = {
                'num_threads': num_threads,
                'requests_per_thread': requests_per_thread,
                'total_requests': num_threads * requests_per_thread,
                'successful_requests': successful_requests,
                'total_time_seconds': total_time,
                'throughput_rps': throughput_rps,
                'mean_latency_ms': statistics.mean(all_latencies),
                'p95_latency_ms': sorted(all_latencies)[int(len(all_latencies) * 0.95)],
                'under_200ms': sum(1 for l in all_latencies if l < 200) / len(all_latencies) * 100
            }
            
            self.results['throughput_tests'].append(results)
            
            print(f"\nThroughput Results:")
            print(f"Successful Requests: {successful_requests}/{num_threads * requests_per_thread}")
            print(f"Total Time: {total_time:.1f}s")
            print(f"Throughput: {throughput_rps:.1f} requests/second")
            print(f"Mean Latency: {results['mean_latency_ms']:.1f}ms")
            print(f"P95 Latency: {results['p95_latency_ms']:.1f}ms")
            print(f"Under 200ms: {results['under_200ms']:.1f}%")
            
            return results
        
        return None
    
    def monitor_resource_usage(self, duration_seconds: int = 60):
        """Monitor CPU and memory usage"""
        print(f"Monitoring resource usage for {duration_seconds} seconds...")
        
        cpu_usage = []
        memory_usage = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            cpu_usage.append(cpu_percent)
            memory_usage.append(memory_info.percent)
            
            print(f"CPU: {cpu_percent:.1f}%, Memory: {memory_info.percent:.1f}%")
        
        results = {
            'duration_seconds': duration_seconds,
            'avg_cpu_percent': statistics.mean(cpu_usage),
            'max_cpu_percent': max(cpu_usage),
            'avg_memory_percent': statistics.mean(memory_usage),
            'max_memory_percent': max(memory_usage)
        }
        
        self.results['resource_usage'].append(results)
        
        print(f"\nResource Usage Results:")
        print(f"Average CPU: {results['avg_cpu_percent']:.1f}%")
        print(f"Max CPU: {results['max_cpu_percent']:.1f}%")
        print(f"Average Memory: {results['avg_memory_percent']:.1f}%")
        print(f"Max Memory: {results['max_memory_percent']:.1f}%")
        
        return results
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        print("Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"Health Status: {health_data['status']}")
                print(f"Model Loaded: {health_data['model_loaded']}")
                return health_data
            else:
                print(f"Health check failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"Health check error: {e}")
            return None
    
    def run_full_performance_test(self):
        """Run complete performance test suite"""
        print("=" * 60)
        print("GuardianAI ML Service Performance Test")
        print("=" * 60)
        
        # Test health first
        health = self.test_health_endpoint()
        if not health or health.get('status') != 'healthy':
            print("Service not healthy, aborting tests")
            return
        
        print("\n" + "=" * 60)
        
        # Test single request latency
        self.test_single_request_latency(100)
        
        print("\n" + "=" * 60)
        
        # Test concurrent throughput
        self.test_concurrent_throughput(10, 20)
        
        print("\n" + "=" * 60)
        
        # Monitor resource usage during load
        print("Starting resource monitoring during load test...")
        
        # Start resource monitoring in background
        import threading
        resource_thread = threading.Thread(
            target=self.monitor_resource_usage,
            args=(30,)
        )
        resource_thread.start()
        
        # Run concurrent load
        time.sleep(5)  # Let monitoring start
        self.test_concurrent_throughput(5, 10)
        
        resource_thread.join()
        
        print("\n" + "=" * 60)
        print("Performance Test Complete")
        print("=" * 60)
        
        # Save results
        with open('performance_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("Results saved to performance_results.json")

def main():
    """Run performance profiling"""
    profiler = PerformanceProfiler()
    profiler.run_full_performance_test()

if __name__ == "__main__":
    main()
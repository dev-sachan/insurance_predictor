# test_simple.py - Fixed test script with working URLs
import asyncio
import aiohttp
import time
import json

class SimpleHackerXTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def test_health_check(self):
        """Test system health"""
        print("🏥 Testing health check...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Health Status: {data['status']}")
                        print(f"📊 System: {data['system']}")
                        print(f"📋 Documents Cached: {data.get('documents_cached', 0)}")
                        return True
                    else:
                        print(f"❌ Health check failed: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_basic_functionality(self):
        """Test basic HackerX endpoint functionality"""
        print("🧪 Testing basic functionality...")
        
        # Using the working URL from your test
        test_payload = {
            "documents": [
                "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
            ],
            "questions": [
                "What is the grace period for premium payment?",
                "What is the waiting period for pre-existing diseases?",
                "Does this policy cover maternity expenses?"
            ]
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/hackerx/run",
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # Increased timeout
                ) as response:
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Basic test passed in {processing_time:.2f}s")
                        print(f"📝 Answers received: {len(data['answers'])}")
                        
                        for i, answer in enumerate(data['answers'], 1):
                            print(f"   Q{i}: {answer[:100]}...")
                        
                        return data, processing_time
                    else:
                        print(f"❌ Basic test failed: {response.status}")
                        text = await response.text()
                        print(f"Error: {text}")
                        return None, processing_time
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Basic test error: {e}")
            return None, processing_time
    
    async def test_all_insurance_questions(self):
        """Test with all 10 insurance questions from HackerX"""
        print("🏥 Testing all 10 insurance questions...")
        
        comprehensive_payload = {
            "documents": [
                "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
            ],
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                "What is the waiting period for pre-existing diseases (PED) to be covered?",
                "Does this policy cover maternity expenses, and what are the conditions?",
                "What is the waiting period for cataract surgery?",
                "Are the medical expenses for an organ donor covered under this policy?",
                "What is the No Claim Discount (NCD) offered in this policy?",
                "Is there a benefit for preventive health check-ups?",
                "How does the policy define a 'Hospital'?",
                "What is the extent of coverage for AYUSH treatments?",
                "Are there any sub-limits on room rent and ICU charges for Plan A?"
            ]
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/hackerx/run",
                    json=comprehensive_payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # Increased timeout
                ) as response:
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ All questions test passed in {processing_time:.2f}s")
                        print(f"📝 Questions processed: {len(data['answers'])}")
                        
                        # Show all answers with question numbers
                        for i, (question, answer) in enumerate(zip(comprehensive_payload['questions'], data['answers']), 1):
                            print(f"\n❓ Q{i}: {question}")
                            if "No documents available" not in answer:
                                print(f"✅ A{i}: {answer}")
                            else:
                                print(f"⚠️ A{i}: {answer}")
                        
                        if data.get('metadata'):
                            metadata = data['metadata']
                            print(f"\n📊 Processing Time: {metadata.get('processing_time_seconds', 'N/A')}s")
                            print(f"📋 Documents Processed: {metadata.get('documents_processed', 'N/A')}")
                            print(f"📥 Successful Downloads: {metadata.get('successful_document_downloads', 'N/A')}")
                        
                        return data, processing_time
                    else:
                        print(f"❌ Comprehensive test failed: {response.status}")
                        text = await response.text()
                        print(f"Error: {text}")
                        return None, processing_time
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Comprehensive test error: {e}")
            return None, processing_time
    
    async def test_error_handling(self):
        """Test error handling"""
        print("🚨 Testing error handling...")
        
        test_cases = [
            {
                "name": "Empty documents",
                "payload": {"documents": [], "questions": ["Test question?"]},
                "expected_status": 400
            },
            {
                "name": "Empty questions", 
                "payload": {"documents": ["https://example.com/test.pdf"], "questions": []},
                "expected_status": 400
            },
            {
                "name": "Too many documents",
                "payload": {"documents": ["https://example.com/test.pdf"] * 15, "questions": ["Test?"]},
                "expected_status": 400
            }
        ]
        
        passed_tests = 0
        
        async with aiohttp.ClientSession() as session:
            for test_case in test_cases:
                try:
                    async with session.post(
                        f"{self.base_url}/hackerx/run",
                        json=test_case["payload"],
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        status = response.status
                        expected = test_case["expected_status"]
                        success = status == expected
                        
                        print(f"{'✅' if success else '❌'} {test_case['name']}: Expected {expected}, Got {status}")
                        
                        if success:
                            passed_tests += 1
                        else:
                            # Show response for debugging
                            try:
                                error_text = await response.text()
                                print(f"   Response: {error_text[:100]}...")
                            except:
                                pass
                            
                except Exception as e:
                    print(f"❌ {test_case['name']}: Exception - {e}")
        
        print(f"📊 Error handling tests: {passed_tests}/{len(test_cases)} passed")
        return passed_tests == len(test_cases)
    
    async def test_debug_cache(self):
        """Test cache debug endpoint"""
        print("🔍 Testing cache status...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/debug/cache") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Cache Status: {data['cached_documents']} cached, {data['active_documents']} active")
                        return True
                    else:
                        print(f"⚠️ Cache debug failed: {response.status}")
                        return False
        except Exception as e:
            print(f"⚠️ Cache debug error: {e}")
            return False
    
    async def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🎯 Starting HackerX Test Suite")
        print("=" * 50)
        
        overall_start = time.time()
        
        # Test 1: Health Check
        health_ok = await self.test_health_check()
        if not health_ok:
            print("❌ System not healthy, aborting tests")
            return False
        
        print("-" * 50)
        
        # Test 2: Cache Status
        await self.test_debug_cache()
        
        print("-" * 50)
        
        # Test 3: Basic Functionality
        basic_result, basic_time = await self.test_basic_functionality()
        if not basic_result:
            print("❌ Basic functionality failed")
            return False
        
        # Check if basic test actually processed documents
        basic_has_real_answers = any("No documents available" not in answer 
                                   for answer in basic_result.get('answers', []))
        
        print("-" * 50)
        
        # Test 4: All Insurance Questions (Most Important!)
        comprehensive_result, comprehensive_time = await self.test_all_insurance_questions()
        if not comprehensive_result:
            print("❌ Comprehensive test failed")
            return False
        
        # Check if comprehensive test actually processed documents
        comp_has_real_answers = any("No documents available" not in answer 
                                  for answer in comprehensive_result.get('answers', []))
        
        print("-" * 50)
        
        # Test 5: Error Handling
        error_handling_ok = await self.test_error_handling()
        
        print("-" * 50)
        
        # Test 6: Final Cache Status
        await self.test_debug_cache()
        
        # Final Results
        total_time = time.time() - overall_start
        
        print("-" * 50)
        print("🎉 TEST SUITE COMPLETED")
        print("=" * 50)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"📊 Basic test time: {basic_time:.2f}s")
        print(f"📊 Comprehensive test time: {comprehensive_time:.2f}s")
        print(f"🚨 Error handling: {'✅ Passed' if error_handling_ok else '❌ Failed'}")
        print(f"📄 Document processing: {'✅ Working' if (basic_has_real_answers or comp_has_real_answers) else '❌ Not working'}")
        
        # Success criteria
        documents_working = basic_has_real_answers or comp_has_real_answers
        
        if comprehensive_result and basic_result and documents_working:
            print("\n🎉 ALL CRITICAL TESTS PASSED!")
            print("🚀 System is ready for HackerX submission!")
            print(f"🔗 Submit this webhook URL: {self.base_url}/hackerx/run")
            return True
        elif comprehensive_result and basic_result:
            print("\n⚠️ TESTS PASSED BUT DOCUMENTS NOT PROCESSING!")
            print("🔧 Fix document processing before submission")
            return False
        else:
            print("\n❌ Some critical tests failed")
            return False

# Quick test function for immediate verification
async def quick_test():
    """Quick test to verify system is working"""
    base_url = "http://localhost:8000"
    
    print("🚀 Quick System Test")
    print("-" * 30)
    
    # Quick health check
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ System Status: {data['status']}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Quick functionality test with working URL
    try:
        test_payload = {
            "documents": ["https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"],
            "questions": ["What is the grace period for premium payment?"]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/hackerx/run", json=test_payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data['answers'][0]
                    print(f"✅ Test Answer: {answer[:80]}...")
                    
                    if "No documents available" not in answer:
                        print("🎯 System is working correctly!")
                        return True
                    else:
                        print("⚠️ System working but documents not processing!")
                        return False
                else:
                    print(f"❌ Test failed: {response.status}")
                    text = await response.text()
                    print(f"Error: {text[:200]}...")
                    return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# Main execution
async def main():
    import sys
    
    # Check if server URL is provided
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"🎯 Testing HackerX system at: {server_url}")
    print(f"🕐 Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if user wants quick test or full suite
    if len(sys.argv) > 2 and sys.argv[2] == "quick":
        success = await quick_test()
    else:
        tester = SimpleHackerXTester(server_url)
        success = await tester.run_full_test_suite()
    
    if success:
        print("\n🏆 READY FOR HACKERX SUBMISSION! 🏆")
    else:
        print("\n❌ Fix issues before submission")

if __name__ == "__main__":
    asyncio.run(main())
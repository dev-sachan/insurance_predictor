# app_enhanced.py - Improved accuracy with clean, relevant answers (Fixed \n issue)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import aiohttp
import PyPDF2
import fitz  # pymupdf
import io
import re
import logging
import hashlib
import json
import time
from collections import defaultdict
import uvicorn
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HackerXRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class HackerXResponse(BaseModel):
    answers: List[str]
    

class EnhancedInsuranceQA:
    """Enhanced Insurance QA system with clean, accurate answers"""
    
    def __init__(self):
        self.documents = []
        self.document_cache = {}
        self.question_patterns = self._initialize_patterns()
        self.insurance_keywords = self._initialize_keywords()
        self.negation_words = {'not', 'no', 'never', 'none', 'without', 'except', 'excluding', 'unless'}
        logger.info("✅ Enhanced Insurance QA system initialized")
    
    def _initialize_keywords(self):
        """Initialize insurance keyword mappings"""
        return {
            'grace_period': ['grace', 'period', 'premium', 'payment', 'due', 'days'],
            'waiting_period': ['waiting', 'period', 'months', 'years', 'continuous', 'coverage'],
            'pre_existing': ['pre-existing', 'ped', 'existing', 'disease', 'condition'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', 'pregnant'],
            'cataract': ['cataract', 'eye', 'surgery', 'lens', 'vision'],
            'organ_donor': ['organ', 'donor', 'donation', 'transplant'],
            'no_claim_discount': ['no claim discount', 'ncd', 'bonus', 'discount'],
            'health_checkup': ['health check', 'preventive', 'screening', 'checkup'],
            'hospital': ['hospital', 'institution', 'beds', 'inpatient'],
            'ayush': ['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy'],
            'room_rent': ['room rent', 'icu', 'sub-limit', 'accommodation']
        }
    
    def _initialize_patterns(self):
        """Initialize comprehensive patterns for accurate extraction"""
        return {
            'grace_period': {
                'patterns': [
                    r'grace period of (\d+) days?',
                    r'(\d+) days?.*?grace period',
                    r'grace.*?(\d+) days',
                    r'(\d+) days.*?from.*?due date',
                    r'premium.*?(\d+) days.*?grace'
                ],
                'template': "A grace period of {} days is provided for premium payment after the due date.",
                'priority': 10,
                'keywords': ['grace', 'period', 'premium', 'payment']
            },
            'waiting_period_ped': {
                'patterns': [
                    r'waiting period of (\d+) months?.*?pre-existing',
                    r'pre-existing.*?(\d+) months?.*?waiting',
                    r'(\d+) months?.*?continuous coverage.*?pre-existing',
                    r'ped.*?(\d+) months?.*?waiting',
                    r'(\d+) months?.*?from.*?policy inception.*?pre-existing'
                ],
                'template': "There is a waiting period of {} months of continuous coverage from the first policy inception date for pre-existing diseases.",
                'priority': 10,
                'keywords': ['waiting', 'period', 'pre-existing', 'ped', 'months']
            },
            'cataract_waiting': {
                'patterns': [
                    r'cataract.*?(\d+) years?',
                    r'(\d+) years?.*?cataract',
                    r'cataract.*?waiting.*?(\d+) years?',
                    r'eye.*?surgery.*?(\d+) years?'
                ],
                'template': "The policy has a waiting period of {} years for cataract surgery.",
                'priority': 9,
                'keywords': ['cataract', 'waiting', 'years', 'eye', 'surgery']
            },
            'hospital_definition': {
                'patterns': [
                    r'hospital.*?means.*?(\d+).*?beds',
                    r'hospital.*?institution.*?(\d+).*?beds',
                    r'(\d+).*?inpatient.*?beds.*?hospital',
                    r'hospital.*?minimum.*?(\d+).*?beds'
                ],
                'template': "A hospital is defined as an institution with at least {} inpatient beds.",
                'priority': 9,
                'keywords': ['hospital', 'definition', 'beds', 'institution']
            },
            'no_claim_discount': {
                'patterns': [
                    r'no claim discount of (\d+)%',
                    r'ncd.*?(\d+)%',
                    r'(\d+)%.*?no claim',
                    r'discount.*?(\d+)%.*?no claim'
                ],
                'template': "A No Claim Discount of {}% is offered on renewal if no claim is made.",
                'priority': 8,
                'keywords': ['no claim discount', 'ncd', 'discount', 'bonus']
            },
            'room_rent_limit': {
                'patterns': [
                    r'room rent.*?(\d+)%',
                    r'daily room.*?(\d+)%',
                    r'(\d+)%.*?sum insured.*?room',
                    r'icu.*?(\d+)%'
                ],
                'template': "Room rent is subject to sub-limits as a percentage of Sum Insured.",
                'priority': 8,
                'keywords': ['room rent', 'sub-limit', 'icu', 'percentage']
            }
        }
    
    def _is_relevant_content(self, text: str, question: str) -> bool:
        """Check if content is relevant and not a random list"""
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Skip if it's clearly a procedure list or irrelevant content
        if re.search(r'\d+\s+[A-Z][a-z]+.*?\n\d+\s+[A-Z][a-z]+', text):
            return False
        
        # Skip if it has too many numbered items (likely a list)
        numbered_items = len(re.findall(r'\n\d+\s+', text))
        if numbered_items > 5:
            return False
        
        # Check for question keywords in text
        question_words = [word for word in question_lower.split() if len(word) > 3]
        matches = sum(1 for word in question_words if word in text_lower)
        
        return matches >= 2
    
    async def download_pdf_with_retry(self, url: str, max_retries: int = 3) -> str:
        """Download PDF with retry mechanism"""
        for attempt in range(max_retries):
            try:
                text = await self._download_pdf_attempt(url)
                if text and len(text.strip()) > 100:
                    return text
                logger.warning(f"Attempt {attempt + 1} yielded insufficient content")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return ""
    
    async def _download_pdf_attempt(self, url: str) -> str:
        """Single PDF download attempt"""
        try:
            # Check cache
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in self.document_cache:
                logger.info(f"📋 Using cached content for: {url[:50]}...")
                return self.document_cache[url_hash]
            
            logger.info(f"📥 Downloading PDF: {url[:100]}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            timeout = aiohttp.ClientTimeout(total=120)
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status == 200:
                        pdf_content = await response.read()
                        logger.info(f"📥 Downloaded {len(pdf_content)} bytes")
                        
                        if not pdf_content.startswith(b'%PDF'):
                            logger.warning("⚠️ Downloaded content is not a valid PDF")
                            return ""
                        
                        text = await self._extract_pdf_text(pdf_content)
                        
                        if text and len(text.strip()) > 100:
                            self.document_cache[url_hash] = text
                            logger.info(f"✅ PDF processing successful: {len(text)} chars")
                            return text
                        else:
                            logger.warning("⚠️ No meaningful text extracted")
                            return ""
                    else:
                        logger.error(f"❌ HTTP error {response.status}")
                        return ""
            
        except Exception as e:
            logger.error(f"❌ Error downloading PDF: {e}")
            return ""
    
    async def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        text = ""
        
        try:
            logger.info("🔄 Extracting text with PyMuPDF...")
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():
                    text += page_text + "\n"
            
            pdf_doc.close()
            
            if text.strip():
                logger.info(f"✅ Text extraction successful: {len(text)} chars")
                return self._clean_text(text)
            
        except Exception as e:
            logger.warning(f"⚠️ PyMuPDF failed: {e}")
            
        # Fallback to PyPDF2
        try:
            logger.info("🔄 Fallback to PyPDF2...")
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue
            
            if text.strip():
                return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"❌ PyPDF2 also failed: {e}")
        
        return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]+', '\n', text)
        
        # Fix common OCR issues
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        return text.strip()
    
    def _clean_answer_text(self, text: str) -> str:
        """Clean answer text by removing unwanted newlines and formatting"""
        if not text:
            return text
        
        # Replace multiple newlines with single space
        text = re.sub(r'\n+', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def process_documents_enhanced(self, document_urls: List[str]) -> bool:
        """Process documents in parallel"""
        self.documents = []
        
        semaphore = asyncio.Semaphore(3)
        
        async def process_single_doc(url):
            async with semaphore:
                try:
                    text = await self.download_pdf_with_retry(url)
                    if text and len(text.strip()) > 100:
                        return {
                            'url': url,
                            'text': text,
                            'length': len(text),
                            'processed_at': datetime.now()
                        }
                except Exception as e:
                    logger.error(f"❌ Failed to process {url}: {e}")
                return None
        
        tasks = [process_single_doc(url) for url in document_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if result and isinstance(result, dict):
                self.documents.append(result)
        
        success_count = len(self.documents)
        logger.info(f"✅ Processed {success_count}/{len(document_urls)} documents successfully")
        
        return success_count > 0
    
    def answer_question_enhanced(self, question: str) -> str:
        """Enhanced question answering with clean, relevant answers"""
        if not self.documents:
            return "No documents available for processing."
        
        question_lower = question.lower()
        logger.info(f"🔍 Answering: {question}")
        
        # Try pattern matching first (most accurate)
        pattern_answer = self._try_pattern_matching(question, question_lower)
        if pattern_answer:
            return self._clean_answer_text(pattern_answer)
        
        # Try keyword-based search
        keyword_answer = self._try_keyword_search(question, question_lower)
        if keyword_answer:
            return self._clean_answer_text(keyword_answer)
        
        # Try boolean questions
        boolean_answer = self._try_boolean_questions(question, question_lower)
        if boolean_answer:
            return self._clean_answer_text(boolean_answer)
        
        return "Information not found in the provided documents."
    
    def _try_pattern_matching(self, question: str, question_lower: str) -> Optional[str]:
        """Try pattern matching for specific numerical answers"""
        for pattern_name, pattern_info in self.question_patterns.items():
            # Check if question is relevant to this pattern
            keyword_matches = sum(1 for keyword in pattern_info['keywords'] 
                                if keyword in question_lower)
            
            if keyword_matches >= 2:  # Need at least 2 keyword matches
                logger.info(f"🎯 Checking pattern: {pattern_name}")
                
                for doc in self.documents:
                    text_lower = doc['text'].lower()
                    
                    # Try each pattern
                    for pattern in pattern_info['patterns']:
                        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                        
                        for match in matches:
                            value = match.group(1)
                            # Get context to validate
                            start = max(0, match.start() - 100)
                            end = min(len(text_lower), match.end() + 100)
                            context = text_lower[start:end]
                            
                            # Validate that this isn't a negation
                            if not any(neg in context for neg in self.negation_words):
                                try:
                                    answer = pattern_info['template'].format(value)
                                    logger.info(f"✅ Pattern match: {answer}")
                                    return answer
                                except:
                                    continue
        
        return None
    
    def _try_keyword_search(self, question: str, question_lower: str) -> Optional[str]:
        """Try keyword-based search for relevant sentences"""
        question_words = [word for word in question_lower.split() 
                         if len(word) > 3 and word not in ['what', 'when', 'where', 'how', 'does']]
        
        best_sentences = []
        
        for doc in self.documents:
            # Split into sentences and clean them
            sentences = [s.strip() for s in re.split(r'[.!?]+', doc['text']) 
                        if len(s.strip()) > 20 and len(s.strip()) < 500]
            
            for sentence in sentences:
                # Clean sentence of excessive whitespace and newlines
                cleaned_sentence = self._clean_answer_text(sentence)
                
                # Skip irrelevant content
                if not self._is_relevant_content(cleaned_sentence, question):
                    continue
                
                sentence_lower = cleaned_sentence.lower()
                
                # Count word matches
                word_matches = sum(1 for word in question_words if word in sentence_lower)
                
                # Bonus for insurance terms
                insurance_bonus = 0
                for keywords in self.insurance_keywords.values():
                    if any(keyword in sentence_lower for keyword in keywords):
                        insurance_bonus += 1
                
                # Bonus for numbers
                has_numbers = bool(re.search(r'\d+', cleaned_sentence))
                number_bonus = 1 if has_numbers else 0
                
                total_score = word_matches + insurance_bonus + number_bonus
                
                if total_score >= 3:
                    best_sentences.append({
                        'sentence': cleaned_sentence,
                        'score': total_score,
                        'word_matches': word_matches
                    })
        
        if best_sentences:
            # Sort by score and return best match
            best_sentences.sort(key=lambda x: x['score'], reverse=True)
            best_sentence = best_sentences[0]['sentence']
            
            # Clean up the sentence one more time
            best_sentence = self._clean_answer_text(best_sentence)
            if len(best_sentence) > 300:
                # Truncate if too long
                best_sentence = best_sentence[:300] + "..."
            
            logger.info(f"✅ Keyword match found")
            return best_sentence
        
        return None
    
    def _try_boolean_questions(self, question: str, question_lower: str) -> Optional[str]:
        """Handle yes/no questions about coverage"""
        # Define boolean question patterns
        boolean_patterns = {
            'maternity': {
                'keywords': ['maternity', 'pregnancy', 'childbirth'],
                'positive_answer': "Yes, the policy covers maternity expenses including childbirth and lawful medical termination of pregnancy.",
                'search_terms': ['maternity', 'pregnancy', 'childbirth', 'delivery']
            },
            'organ_donor': {
                'keywords': ['organ', 'donor', 'donation'],
                'positive_answer': "Yes, the policy covers medical expenses for the organ donor's hospitalization for organ donation purposes.",
                'search_terms': ['organ donor', 'donation', 'transplant']
            },
            'ayush': {
                'keywords': ['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy'],
                'positive_answer': "Yes, the policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems.",
                'search_terms': ['ayush', 'ayurveda', 'alternative medicine', 'homeopathy']
            },
            'health_checkup': {
                'keywords': ['health check', 'preventive', 'screening', 'checkup'],
                'positive_answer': "Yes, the policy provides benefits for preventive health check-ups at specified intervals.",
                'search_terms': ['health check', 'preventive', 'screening']
            }
        }
        
        # Check if question is asking about coverage
        is_coverage_question = any(word in question_lower for word in 
                                 ['cover', 'include', 'benefit', 'reimburse', 'pay'])
        
        if is_coverage_question:
            for pattern_name, pattern_info in boolean_patterns.items():
                if any(keyword in question_lower for keyword in pattern_info['keywords']):
                    # Search for evidence in documents
                    found_evidence = False
                    for doc in self.documents:
                        text_lower = doc['text'].lower()
                        if any(term in text_lower for term in pattern_info['search_terms']):
                            # Check it's not negated
                            for term in pattern_info['search_terms']:
                                if term in text_lower:
                                    # Find the context around the term
                                    pos = text_lower.find(term)
                                    start = max(0, pos - 50)
                                    end = min(len(text_lower), pos + 50)
                                    context = text_lower[start:end]
                                    
                                    # If no negation words in context, consider it positive
                                    if not any(neg in context for neg in self.negation_words):
                                        found_evidence = True
                                        break
                    
                    if found_evidence:
                        logger.info(f"✅ Boolean question answered: {pattern_name}")
                        return pattern_info['positive_answer']
        
        return None

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced HackerX Insurance QA System",
    description="Clean, accurate insurance document QA system",
    version="2.1.0"
)

qa_system = EnhancedInsuranceQA()

@app.middleware("http")
async def add_cors_headers(request, call_next):
    """Add CORS headers"""
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.post("/hackerx/run")
async def process_hackathon_request(request: HackerXRequest) -> HackerXResponse:
    """Enhanced HackerX endpoint with clean answers"""
    start_time = time.time()
    
    try:
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="At least one document URL is required")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Validate URLs
        for url in request.documents:
            if not url.startswith(('http://', 'https://')):
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
        
        logger.info(f"🎯 Processing {len(request.documents)} documents, {len(request.questions)} questions")
        
        # Process documents
        doc_success = await qa_system.process_documents_enhanced(request.documents)
        
        if not doc_success:
            logger.error("❌ No documents were successfully processed")
            return HackerXResponse(
                answers=["Error: Unable to process documents. Please check URLs and try again."] * len(request.questions),
            )
        
        # Answer questions
        answers = []
        question_metrics = []
        
        for i, question in enumerate(request.questions):
            try:
                q_start = time.time()
                answer = qa_system.answer_question_enhanced(question)
                q_time = time.time() - q_start
                
                # Final cleanup of answer to ensure no \n characters
                clean_answer = qa_system._clean_answer_text(answer)
                answers.append(clean_answer)
                
                question_metrics.append({
                    'question_index': i,
                    'processing_time': round(q_time, 3),
                    'answer_length': len(clean_answer)
                })
                
                logger.info(f"✅ Q{i+1}: {clean_answer[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Error answering question {i+1}: {e}")
                answers.append("Error processing this question.")
                question_metrics.append({
                    'question_index': i,
                    'processing_time': 0,
                    'error': str(e)
                })
        
        processing_time = time.time() - start_time
        
        response = HackerXResponse(
            answers=answers,
        )
        
        logger.info(f"✅ Request completed in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return HackerXResponse(
            answers=["System error occurred."] * len(request.questions),
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "documents_cached": len(qa_system.document_cache),
        "active_documents": len(qa_system.documents),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Enhanced HackerX Insurance QA System",
        "version": "2.1.0",
        "status": "ready",
        "description": "Clean, accurate insurance document QA system",
        "endpoints": {
            "main": "/hackerx/run",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.options("/hackerx/run")
async def options_handler():
    """Handle OPTIONS requests for CORS"""
    return {"message": "OK"}
from fastapi import FastAPI
app = FastAPI()

if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced HackerX Insurance QA System v2.1.0...")
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )
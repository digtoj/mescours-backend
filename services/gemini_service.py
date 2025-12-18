from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Optional


class GeminiService:
    """
    Service to interact with Google's Gemini API.
    """
    
    def _get_llm(self, api_key: str, temperature: float = 0.7):
        """
        Create a Gemini LLM instance with user's API key.
        
        Interview point: We create a new instance per request because
        each user has their own API key. In a shared-key scenario,
        you'd use a singleton pattern instead.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Fast and cheap
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True  # Gemini quirk
        )
    
    async def generate_summary(self, content: str, api_key: str, language: str = "English") -> dict:
        """
        Generate a summary, key points, and flashcards from course content.
        
        Returns dict with: summary, key_points, flashcards
        """
        llm = self._get_llm(api_key, temperature=0.3)  # Lower temp = more focused
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational assistant. 
            Analyze the following course material and provide:
            1. A concise summary/resume (2-3 paragraphs) capturing the core meaning.
            2. Key points (5-7 bullet points of the most important concepts).
            3. 5 study flashcards (Question/Answer pairs) testing key concepts.
            
            IMPORTANT INSTRUCTIONS:
            - The output MUST be in {language}.
            
            Format your response exactly as:
            SUMMARY:
            [Your summary in {language}]
            
            KEY_POINTS:
            - [Point 1 in {language}]
            - [Point 2 in {language}]
            ...
            
            FLASHCARDS:
            Front: [Question 1]
            Back: [Answer 1]
            
            Front: [Question 2]
            Back: [Answer 2]
            ...
            """),
            ("human", "{content}")
        ])  
        
        chain = prompt | llm
        try:
            response = await chain.ainvoke({
                "content": content[:15000],
                "language": language
            })
            
            # Parse the response
            return self._parse_summary_response(response.content)
            
        except Exception as e:
            print(f"Error in generate_summary: {e}")
            raise e
    
    
    def _parse_summary_response(self, response: str) -> dict:
        """Parse the structured response into a dictionary."""
        result = {
            "summary": "",
            "key_points": [],
            "flashcards": []
        }
        
        try:
            current_section = None
            current_card = {}
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if "SUMMARY:" in line.upper():
                    current_section = "summary"
                elif "KEY_POINTS:" in line.upper():
                    current_section = "key_points"
                elif "FLASHCARDS:" in line.upper():
                    current_section = "flashcards"
                elif current_section == "summary":
                    result["summary"] += line + " "
                elif current_section == "key_points" and line.startswith("-"):
                    result["key_points"].append(line[1:].strip())
                elif current_section == "flashcards":
                    if line.upper().startswith("FRONT:"):
                        if "front" in current_card and "back" in current_card:
                            result["flashcards"].append(current_card)
                            current_card = {}
                        current_card["front"] = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("BACK:"):
                        current_card["back"] = line.split(":", 1)[1].strip()
            
            # Add last card if exists
            if "front" in current_card and "back" in current_card:
                result["flashcards"].append(current_card)
            
            # Clean up
            result["summary"] = result["summary"].strip()
            
            # If nothing parsed (fallback)
            if not result["summary"]:
                result["summary"] = response
                
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            # Return raw text as summary if parsing totally fails
            result["summary"] = response
            
        return result
    
    async def answer_question(
        self, 
        question: str, 
        course_content: str, 
        api_key: str,
        chat_history: Optional[List[dict]] = None
    ) -> str:
        """
        Answer a question based on course content (RAG-style).
        
        Interview point: This is a simple RAG implementation.
        In production, you'd use vector embeddings for retrieval.
        """
        llm = self._get_llm(api_key, temperature=0.5)
        
        # Build context from chat history
        history_text = ""
        if chat_history:
            for msg in chat_history[-5:]:  # Last 5 messages
                role = "Student" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful study assistant. Answer questions based ONLY on the provided course material.

Course Material:
{course_content}

Previous conversation:
{history}

Rules:
1. Only use information from the course material
2. If the answer isn't in the material, say "I couldn't find this in your course material"
3. Be concise but thorough
4. Reference specific concepts from the material when possible
"""),
            ("human", "{question}")
        ])
        
        chain = prompt | llm
        response = await chain.ainvoke({
            "course_content": course_content[:20000],  # Limit to avoid token limits
            "history": history_text,
            "question": question
        })
        
        return response.content


# Singleton instance
gemini_service = GeminiService()
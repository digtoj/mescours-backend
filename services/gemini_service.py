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
    
    async def generate_summary(self, content: str, api_key: str) -> dict:
        """
        Generate a summary with key points from course content.
        
        Returns dict with: summary, key_points, audio_script
        """
        llm = self._get_llm(api_key, temperature=0.3)  # Lower temp = more focused
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational assistant. 
            Analyze the following course material and provide:
            1. A concise summary (2-3 paragraphs)
            2. Key points (5-7 bullet points of the most important concepts)
            3. An audio script (a natural, spoken version of the summary for text-to-speech)
            
            Format your response as:
            SUMMARY:
            [Your summary here]
            
            KEY_POINTS:
            - Point 1
            - Point 2
            ...
            
            AUDIO_SCRIPT:
            [Natural spoken version]
            """),
            ("human", "{content}")
        ])
        
        chain = prompt | llm
        response = await chain.ainvoke({"content": content[:15000]})  # Limit content size
        
        # Parse the response
        return self._parse_summary_response(response.content)
    
    def _parse_summary_response(self, response: str) -> dict:
        """Parse the structured response into a dictionary."""
        result = {
            "summary": "",
            "key_points": [],
            "audio_script": ""
        }
        
        sections = response.split("\n\n")
        current_section = None
        
        for line in response.split("\n"):
            if "SUMMARY:" in line:
                current_section = "summary"
            elif "KEY_POINTS:" in line:
                current_section = "key_points"
            elif "AUDIO_SCRIPT:" in line:
                current_section = "audio_script"
            elif current_section == "summary":
                result["summary"] += line + " "
            elif current_section == "key_points" and line.strip().startswith("-"):
                result["key_points"].append(line.strip()[1:].strip())
            elif current_section == "audio_script":
                result["audio_script"] += line + " "
        
        # Clean up
        result["summary"] = result["summary"].strip()
        result["audio_script"] = result["audio_script"].strip()
        
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
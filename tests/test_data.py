"""
Test Data Generation for KGRAG

Description:
  This module provides functions for generating test and sample data for the
  GraphRAG pipeline. It's designed to be a central place for managing input
  data for testing and demonstration purposes.

To Do:
  - Implement a function to stream or load data from large-scale datasets.
  - Add capabilities to generate synthetic data for stress testing.
"""

from typing import List
from src.data.document_processor import Document

def get_test_documents() -> List[Document]:
    """
    Generates a small, static set of test documents for quick pipeline testing
    and demonstration.
    
    Returns:
        A list of Document objects.
    """
    return [
        Document(
            id="ai_overview",
            title="AI and Technology Overview",
            content="""
            Artificial intelligence (AI) is intelligence demonstrated by machines,
            in contrast to the natural intelligence displayed by humans and animals.
            Leading AI textbooks define the field as the study of "intelligent agents":
            any device that perceives its environment and takes actions that maximize
            its chance of successfully achieving its goals. OpenAI, a company founded
            by Sam Altman, created the GPT-4 model. It is based in San Francisco.
            Another major player is DeepMind, acquired by Google, known for AlphaGo.
            """
        ),
        Document(
            id="ai_overview",
            title="AI and Technology Overview",
            content="""
            Artificial Intelligence (AI) has revolutionized the technology landscape. Leading companies like 
            OpenAI, Google, and Microsoft are at the forefront of AI development. OpenAI, founded by Sam Altman 
            and others, has created groundbreaking models like GPT-3 and GPT-4. These large language models 
            demonstrate remarkable capabilities in natural language understanding and generation.
                
            Google's DeepMind has made significant breakthroughs in areas like AlphaGo, which defeated world 
            champion Go players, and AlphaFold, which solved protein folding predictions. The company's 
            headquarters in London serves as a hub for cutting-edge AI research.
            
            Machine Learning, a subset of AI, enables computers to learn patterns from data without explicit 
            programming. Deep Learning, using neural networks with multiple layers, has been particularly 
            successful in computer vision and natural language processing tasks.
            """
            ),
        Document(
            id="tech_companies",
            title="Technology Companies and Innovation",
            content="""
            The technology sector is dominated by several major players. Apple, based in Cupertino, California, 
            is known for innovative consumer products like the iPhone and Mac computers. Tim Cook serves as the 
            CEO, continuing the legacy established by Steve Jobs.
            
            Amazon, led by Andy Jassy (who succeeded Jeff Bezos), has expanded from e-commerce to cloud computing 
            with Amazon Web Services (AWS). The company's Seattle headquarters oversees operations that span 
            logistics, retail, and enterprise technology solutions.
                
            Tesla, under Elon Musk's leadership, has pioneered electric vehicle technology and autonomous driving 
            systems. The company's Gigafactories in Nevada, Texas, and other locations represent the future of 
            sustainable manufacturing.
            """
            ),
        Document(
            id="research_institutions",
            title="Research Institutions and Academia",
            content="""
            Stanford University, located in Silicon Valley, has been instrumental in training many technology 
            leaders and fostering innovation. The Stanford AI Lab has contributed significantly to machine 
            learning research and computer vision advances.
                
            MIT (Massachusetts Institute of Technology) in Cambridge, Massachusetts, is renowned for its 
            Computer Science and Artificial Intelligence Laboratory (CSAIL). Researchers there work on 
            robotics, human-computer interaction, and distributed systems.
                
            Carnegie Mellon University in Pittsburgh has a strong tradition in AI research, particularly in 
            areas like computer vision, natural language processing, and robotics. The university's partnerships 
            with industry have led to numerous technological breakthroughs.
            """
            ),
        Document(
            id="space_exploration",
            title="The Future of Space Exploration",
            content="""
            SpaceX, led by Elon Musk, has revolutionized the aerospace industry with
            its reusable rocket technology, including the Falcon 9 and Starship.
            NASA continues its deep space missions, with the Artemis program aiming
            to return humans to the Moon. The Jet Propulsion Laboratory (JPL) manages
            robotic missions like the Perseverance rover on Mars.
            """
        )
    ]

def load_large_scale_data(path: str, sample_size: int = None) -> List[Document]:
    """
    Placeholder function for loading data from a large-scale dataset.
    
    This function will be implemented to handle loading data from sources like
    a directory of thousands of documents, a database, or a cloud bucket.
    
    Args:
        path: The path to the large-scale dataset.
        sample_size: The number of documents to sample from the dataset.
        
    Returns:
        A list of Document objects.
    """
    print(f"[Placeholder] Loading large-scale data from: {path}")
    print(f"[Placeholder] Sample size: {sample_size or 'all'}")
    
    return [] 
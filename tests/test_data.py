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
    documents = [
            Document(
                id="ai_companies",
                title="AI Companies and Technology",
                content="""
                OpenAI is an artificial intelligence research company founded by Sam Altman, Elon Musk, and others in San Francisco in 2015.
                The company has developed several breakthrough AI models including GPT-3, GPT-4, and ChatGPT, which are large language models
                capable of understanding and generating human-like text. OpenAI's mission is to ensure that artificial general intelligence
                benefits all of humanity. The company has also created DALL-E, an AI system that can generate images from text descriptions.

                Google, led by CEO Sundar Pichai, has developed competing AI models like PaLM, Bard, and Gemini. Google's DeepMind division,
                headquartered in London, has made significant breakthroughs including AlphaGo, which defeated world champion Go players,
                and AlphaFold, which solved protein folding predictions. DeepMind was founded by Demis Hassabis and acquired by Google in 2014.

                Microsoft, under CEO Satya Nadella, has integrated OpenAI's technology into their products through a strategic partnership.
                Microsoft has invested billions in OpenAI and integrated GPT models into Microsoft Office, Bing search, and Azure cloud services.
                """
            ),
            Document(
                id="tech_leaders",
                title="Technology Leaders and Innovation",
                content="""
                Elon Musk, the CEO of Tesla and SpaceX, has been a major figure in technology innovation. Tesla, headquartered in Austin, Texas,
                has revolutionized the electric vehicle industry with models like the Model S, Model 3, and Model Y. The company's Gigafactories
                in Nevada, Texas, and other locations represent the future of sustainable manufacturing.

                Tim Cook serves as the CEO of Apple, continuing the legacy established by Steve Jobs. Apple, based in Cupertino, California,
                is known for innovative consumer products like the iPhone, iPad, and Mac computers. The company has also ventured into
                artificial intelligence with Siri and machine learning capabilities.

                Jeff Bezos founded Amazon in Seattle and served as CEO before transitioning to Executive Chairman. Amazon has expanded from
                e-commerce to cloud computing with Amazon Web Services (AWS), which has become a dominant force in cloud infrastructure.
                Andy Jassy, former head of AWS, now serves as Amazon's CEO.
                """
            ),
            Document(
                id="research_institutions",
                title="Research Institutions and Academia",
                content="""
                Stanford University, located in Silicon Valley, has been instrumental in training many technology leaders and fostering innovation.
                The Stanford AI Lab has contributed significantly to machine learning research, computer vision, and natural language processing.
                Notable alumni include the founders of Google, Yahoo, and many other tech companies.

                MIT (Massachusetts Institute of Technology) in Cambridge, Massachusetts, is renowned for its Computer Science and Artificial
                Intelligence Laboratory (CSAIL). Researchers there work on robotics, human-computer interaction, and distributed systems.
                MIT has produced numerous technology entrepreneurs and continues to be at the forefront of AI research.

                Carnegie Mellon University in Pittsburgh has a strong tradition in AI research, particularly in computer vision,
                natural language processing, and robotics. The university's partnerships with industry have led to numerous technological
                breakthroughs and successful startups.
                """
            )
        ]

    return documents

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
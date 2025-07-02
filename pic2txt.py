#!pip install transformers chromadb pillow sentence-transformers
#!pip install jieba
#from google.colab import drive
#drive.mount('/content/drive')
#!ls -l /content/drive/MyDrive/assert/train_images/


import chromadb
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
# ========== 初始化模型 ==========
# 使用中文优化的CLIP版本
clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
clip_processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# 初始化DeepSeek模型（文本处理） 暂时用jieba代替
#from transformers import AutoTokenizer, AutoModelForCausalLM
#llm_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
#llm_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# ========== 步骤1：LLM分词处理 ==========
# def semantic_segmentation(text):
#     """
#     将语句拆分为有意义的词/短语
#     返回格式：["短语1", "短语2", ...]
#     """
#     prompt = f"""
#     请将以下文本拆分为独立语义单元，每个单元必须是完整的名词短语或动词短语，
#     输出为JSON列表，不要解释：

#     比如我给你一个："一只橘猫在阳光下的窗台上打盹"
#     输出结果为：["一只橘猫", "阳光下", "窗台上", "打盹"]

#     待处理文本：{text}
#     """

#     inputs = llm_tokenizer(prompt, return_tensors="pt")
#     outputs = llm_model.generate(**inputs, max_new_tokens=100)
#     response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # 提取JSON格式结果
#     start_idx = response.find('[')
#     end_idx = response.find(']') + 1
#     return eval(response[start_idx:end_idx])

import jieba 
seg_list = jieba.cut('穿红色连衣裙的女孩在公园长椅上看书', cut_all=True)
print(list(seg_list))

from PIL import Image
# ========== 步骤2：向量化处理 ==========
def encode_data(image_path, text=None):
    """
    处理单条数据（图片+文本）
    返回：{
        "image_vec": 图片向量, 
        "text_phrases": 分词结果, 
        "phrase_vecs": [短语向量1, ...]
    }
    """
    # 处理图像
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    image_vec = clip_model.get_image_features(**inputs).detach().numpy()

    # 处理文本
    if text:
        # phrases = semantic_segmentation(text)
        phrases = list(jieba.cut(text))
        phrase_vecs = []
        for phrase in phrases:
            text_inputs = clip_processor(text=phrase, return_tensors="pt", padding=True)
            text_vec = clip_model.get_text_features(**text_inputs).detach().numpy()
            phrase_vecs.append(text_vec.flatten().tolist())
    else:
        phrases, phrase_vecs = [], []

    return {
        "image_vec": image_vec.flatten().tolist(),
        "text_phrases": phrases,
        "phrase_vecs": phrase_vecs
    }

# ========== 步骤3：构建向量数据库 ==========
client = chromadb.Client()
client.delete_collection("multimodal_db")
collection_db = client.create_collection("multimodal_db")
# 添加数据示例
data = [
    {"id": "1", "path": "/content/drive/MyDrive/assert/train_images/dog_lawn_yellow.jpeg", "text": "一只大黄狗在草坪上奔跑"},
    {"id": "2", "path": "/content/drive/MyDrive/assert/train_images/dog_lawn_black.jpeg", "text": "一只大黑狗在草坪上趴着"},
    {"id": "3", "path": "/content/drive/MyDrive/assert/train_images/cat_1.jpeg", "text": "一只猫"},
]
for item in data:
    encoded = encode_data(item["path"], item["text"])

    # 存储图片向量
    collection_db.add(
        ids=[f"{item['id']}_image"],
        embeddings=[encoded["image_vec"]],
        metadatas=[{
            "type": "image", 
            "image_path": item["path"], 
            "source": item["id"]
            }]
    )

    # 存储文本短语向量
    for i, (phrase, vec) in enumerate(zip(encoded["text_phrases"], encoded["phrase_vecs"])):
        print(f'第{i}个 : {phrase} : {len(vec)}')
        collection_db.add(
            ids=[f"{item['id']}_text_{i}"],
            embeddings=[vec],
            metadatas=[{
                "type": "text", 
                "phrase": phrase,
                "source": item["id"]
            }]
        )
print("数据库构建完成，image总条目: %s " % (collection_db.count()))

# ========== 步骤4：找到数据库中最像的两张图片 ==========
test_image_path = '/content/drive/MyDrive/assert/train_images/cat_1.jpeg'
test_data = encode_data(test_image_path)
results = collection_db.query(
    query_embeddings=[test_data["image_vec"]],
    where={"type": "image"},
    n_results=2,
    include=["metadatas"]
)

# ========== 步骤5：找到图片拆分后的所有文本并按照embeddings 排序  ==========
source_ids = list(set([item['source'] for item in results["metadatas"][0]]))
image_paths = list(set([item['image_path'] for item in results["metadatas"][0]]))

print(source_ids)
where_condition = {
    "$and": [
        {"source": {"$in": source_ids}},
        {"type": "text"}
    ]
}
phrase_results = collection_db.get(
    #query_embeddings=[test_data["image_vec"]],
    where=where_condition,
    include=["metadatas"]
)

# phrases = list(set([item['phrase'] for item in phrase_results["metadatas"][0]]))
print(phrase_results)

# ============ 步骤6: 使用图文检索找到图片与文字最相似TopN的 ========
import time
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import os
from typing import Union, List, Dict, Tuple
from abc import ABC, abstractmethod

class EmbeddingProcessor(ABC):
    """嵌入向量处理器抽象基类"""
    
    @abstractmethod
    def process(self, input_data: Union[str, Image.Image, List[str]]) -> np.ndarray:
        """处理输入数据并返回嵌入向量"""
        pass


class ImageProcessor(EmbeddingProcessor):
    """图像处理器"""
    
    def __init__(self, model: CLIPModel, processor: CLIPProcessor):
        self.model = model
        self.processor = processor
    
    def process(self, input_data: Union[str, Image.Image]) -> np.ndarray:
        """处理图像输入"""
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Image file {input_data} not found")
            image = Image.open(input_data)
        elif isinstance(input_data, Image.Image):
            image = input_data
        else:
            raise TypeError("input_data must be image path or PIL Image")
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            features = self.model.get_image_features(inputs.pixel_values)[0]
            return features.numpy().astype(np.float32)


class TextProcessor(EmbeddingProcessor):
    """文本处理器"""
    
    def __init__(self, model: CLIPModel, processor: CLIPProcessor):
        self.model = model
        self.processor = processor
    
    def process(self, input_data: Union[str, List[str]]) -> np.ndarray:
        """处理文本输入"""
        with torch.no_grad():
            inputs = self.processor(text=input_data, return_tensors="pt", 
                                  padding=True, truncation=True)
            features = self.model.get_text_features(inputs.input_ids)
            
            if isinstance(input_data, str):
                return features[0].numpy().astype(np.float32)
            else:
                return features.numpy().astype(np.float32)


class CLIPEmbeddingExtractor:
    """CLIP嵌入向量提取器主类"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._image_processor = None
        self._text_processor = None
        
    def _load_model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """懒加载模型和处理器"""
        if self._model is None or self._processor is None:
            print(f"Loading CLIP model: {self.model_name}...")
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            print("Model loaded successfully!")
        return self._model, self._processor
    
    def _get_image_processor(self) -> ImageProcessor:
        """获取图像处理器"""
        if self._image_processor is None:
            model, processor = self._load_model()
            self._image_processor = ImageProcessor(model, processor)
        return self._image_processor
    
    def _get_text_processor(self) -> TextProcessor:
        """获取文本处理器"""
        if self._text_processor is None:
            model, processor = self._load_model()
            self._text_processor = TextProcessor(model, processor)
        return self._text_processor
    
    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """获取图像嵌入向量"""
        start_time = time.time()
        processor = self._get_image_processor()
        embedding = processor.process(image_input)
        
        processing_time = time.time() - start_time
        print(f'Processed image in {processing_time:.4f} seconds')
        print(f'Vector dimension: {embedding.shape[0]}')
        
        return embedding
    
    def get_text_embedding(self, text_input: Union[str, List[str]]) -> np.ndarray:
        """获取文本嵌入向量"""
        start_time = time.time()
        processor = self._get_text_processor()
        embedding = processor.process(text_input)
        
        processing_time = time.time() - start_time
        print(f'Processed text in {processing_time:.4f} seconds')
        print(f'Vector dimension: {embedding.shape[-1] if len(embedding.shape) > 1 else embedding.shape[0]}')
        
        return embedding


class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量之间的余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量之间的欧氏距离"""
        return np.linalg.norm(vec1 - vec2)
    
    def calculate_similarities(self, query_vector: np.ndarray, 
                             candidate_vectors: Dict[str, np.ndarray],
                             metric: str = 'cosine') -> List[Tuple[str, float]]:
        """计算查询向量与候选向量集合的相似度"""
        similarities = []
        
        for name, vector in candidate_vectors.items():
            if metric == 'cosine':
                sim = self.cosine_similarity(query_vector, vector)
            elif metric == 'euclidean':
                sim = -self.euclidean_distance(query_vector, vector)  # 负值使得距离越小相似度越高
            else:
                raise ValueError("metric must be 'cosine' or 'euclidean'")
            
            similarities.append((name, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)


class ImageDisplayer:
    """图像显示器"""
    
    @staticmethod
    def display_image(image_path: str, title: str = None, figsize: Tuple[int, int] = (5, 5)):
        """显示图像并添加标题"""
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found")
            return
            
        img = Image.open(image_path)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        if title:
            plt.title(title, fontsize=10)
        plt.axis('off')
        plt.show()


class CLIPMultimodalDemo:
    """CLIP多模态演示主类"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        self.extractor = CLIPEmbeddingExtractor(model_name)
        self.similarity_calculator = SimilarityCalculator()
        self.image_displayer = ImageDisplayer()
        
    def process_images(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """处理图像列表并返回嵌入向量字典"""
        image_embeddings = {}
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image file {img_path} not found. Skipping...")
                continue
                
            print(f"\nProcessing image: {img_path}")
            self.image_displayer.display_image(img_path)
            
            try:
                embedding = self.extractor.get_image_embedding(img_path)
                image_embeddings[img_path] = embedding
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return image_embeddings
    
    def process_texts(self, text_queries: List[str]) -> Dict[str, np.ndarray]:
        """处理文本列表并返回嵌入向量字典"""
        text_embeddings = {}
        
        for text in text_queries:
            print(f"\nProcessing text: '{text}'")
            try:
                embedding = self.extractor.get_text_embedding(text)
                text_embeddings[text] = embedding
            except Exception as e:
                print(f"Error processing text '{text}': {e}")
                
        return text_embeddings
    
    def analyze_image_text_similarity(self, image_embeddings: Dict[str, np.ndarray], 
                                    text_embeddings: Dict[str, np.ndarray], k: int):
        """分析图像-文本相似度"""
        print("\n" + "="*50)
        print("Image-Text Similarity Results")
        print("="*50)
        
        for img_path, img_vec in image_embeddings.items():
            print(f"\nResults for image: {img_path}")
            self.image_displayer.display_image(img_path)
            
            similarities = self.similarity_calculator.calculate_similarities(
                img_vec, text_embeddings
            )
            
            print("Top matching texts:")
            for i, (text, sim) in enumerate(similarities[:k]):
                print(f"{i+1}. '{text}' - similarity: {sim:.4f}")
    
    def analyze_text_text_similarity(self, text_embeddings: Dict[str, np.ndarray]):
        """分析文本-文本相似度"""
        print("\n" + "="*50)
        print("Text-Text Similarity Results")
        print("="*50)
        
        if not text_embeddings:
            print("No text embeddings available")
            return
        
        # 选择第一个文本作为查询
        query_text = list(text_embeddings.keys())[0]
        query_vec = text_embeddings[query_text]
        
        print(f"\nSimilarity to query: '{query_text}'")
        
        # 创建候选文本字典（排除查询文本）
        candidate_texts = {k: v for k, v in text_embeddings.items() if k != query_text}
        
        similarities = self.similarity_calculator.calculate_similarities(
            query_vec, candidate_texts
        )
        
        for i, (text, sim) in enumerate(similarities):
            print(f"{i+1}. '{text}' - similarity: {sim:.4f}")
    
    def run_demo(self, image_paths: List[str], text_queries: List[str]):
        """运行完整演示"""
        print("="*50)
        print("CLIP Multimodal Embedding Demo")
        print("="*50)
        
        # 处理图像和文本
        image_embeddings = self.process_images(image_paths)
        text_embeddings = self.process_texts(text_queries)
        
        # 分析相似度 最相似的钱4个
        if image_embeddings and text_embeddings:
            self.analyze_image_text_similarity(image_embeddings, text_embeddings, 5)
        
        if len(text_embeddings) > 1:
            self.analyze_text_text_similarity(text_embeddings)


def main():
    """主函数"""
    # 示例图像路径
    image_paths = [
        '/content/drive/MyDrive/assert/train_images/cat_1.jpeg'
    ]
    
    # 示例文本描述
    text_queries = [
        "一只"
        ,"大"
        ,"黑狗"
        ,"在"
        ,"草坪"
        ,"上"
        ,"趴着"
        ,"猫"
    ]
    
    # 创建演示实例并运行
    demo = CLIPMultimodalDemo()
    demo.run_demo(image_paths, text_queries)

if __name__ == "__main__":
    main()
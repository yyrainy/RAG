import config
import os
import traceback
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder



#embending初始化单例模式
class EmbeddingSingleton:
    instance = None
    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu", "local_files_only": True}
            )
        return cls.instance

def get_embeddings():
    return EmbeddingSingleton.get_instance()

#加载文档并切分----->返回切分后的文档块:doc_splits
def load_and_split_pdf(pdf_file_path):
    """

    :param pdf_file_path:上传文档
    :return: 切块后的文档
    """
    file_pos=os.path.splitext(pdf_file_path)[-1].lower()
    suppotr_pos=[".pdf",".txt",".docx",".doc"]
    if file_pos not in suppotr_pos:
        raise ValueError(f"不支持的文件类型！仅支持{suppotr_pos}，当前文件：{file_pos}")
    loader=None
    try:
        if file_pos == ".pdf":
            loader = PyPDFLoader(pdf_file_path)
        elif file_pos == ".txt":
            loader=TextLoader(pdf_file_path)
        elif file_pos == ".docx":
            loader=Docx2txtLoader(pdf_file_path)
        documents = loader.load()
        if not documents:
            raise ValueError("文件为空或无法解析")
        # 递归分块（按语义分割，优先级：空行（段落）>换行>句号>空格）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", " "]
        )
        doc_splits = text_splitter.split_documents(documents)
        # 给每个分块添加元数据（用于溯源，加分亮点）
        for i, doc in enumerate(doc_splits):
            doc.metadata["chunk_id"] = i
            doc.metadata["source"] = os.path.basename(pdf_file_path)
            doc.metadata["file_type"] = file_pos[1:]
        print(f"文档处理完成：{len(doc_splits)}个文本块")
        return doc_splits
    except Exception as e:
        raise ValueError(f"文档处理失败：{str(e)}")

#创建向量库--------->vectordb
def build_vector_db(doc_splits, persist_dir="./faiss_db_cpu"):
    """

    :param doc_splits: 切分的文档
    :param persist_dir: 本地向量库
    :return: 获取向量
    """
    embeddings=get_embeddings()
    faiss_address=os.path.join(config.PERSIST_DIR,"index.faiss")
    if(os.path.exists(faiss_address)):
        vectordb=FAISS.load_local(config.PERSIST_DIR,embeddings,allow_dangerous_deserialization=True)# 必须为True
        vectordb.add_documents(doc_splits)
    else:
        # 构建FAISS向量库
        vectordb = FAISS.from_documents(
            documents=doc_splits,
            embedding=embeddings
        )
    # 持久化FAISS向量库
    vectordb.save_local(persist_dir)
    print("向量库构建完成（FAISS+纯CPU）")
    return vectordb

#数据检索---------->context_with_source, final_docs
def enhanced_retrieval(vectordb, question):
    """
    只需要1就可以完成相关数据检索（粗排），但是第2步的是精排
    :param vectordb: 持久化的向量
    :param question: 要问的问题
    :return: 整合的数据，元数据
    """
    # 1. 向量粗排（语义匹配）
    retriever = vectordb.as_retriever(search_kwargs={"k":config.TOP_K})
    raw_docs = retriever.invoke(question)
    #*********raw_docs结果是字典：page_content,metadata************
    # 2. 重排序（提升相关性，过滤无关内容）
    reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")  # 强制CPU(本人电脑是FW)
    # 构造重排序输入：(问题, 文本)
    rerank_inputs = [(question, doc.page_content) for doc in raw_docs]
    # 计算相关性分数
    scores=reranker.predict(rerank_inputs)
    # 按分数排序，取Top3
    doc_score_pairs = list(zip(raw_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    final_docs = [pair[0] for pair in doc_score_pairs[:config.TOP_N]]

    # 拼接上下文+溯源信息
    context_with_source = ""
    for i, doc in enumerate(final_docs):
        source = doc.metadata.get("source", "未知文档")
        page = doc.metadata.get("page", "未知页码")
        context_with_source += f"【参考信息{i + 1}】\n内容：{doc.page_content}\n来源：{source} 第{page}页\n\n"
    return context_with_source, final_docs

#加载已有的向量库--------->vectordb
def load_exist_vector_db(persist_dir="./faiss_db_cpu"):
    try:
        embeddings=get_embeddings()
        # 加载本地已保存的faiss库
        vectordb = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("成功加载本地已保存的向量库")
        return vectordb
    except:
        print("未找到本地向量库，请先上传文档")
        return None

#删除向量库中的指定向量
def delete_file_from_vector_db(pdf_filename):
    """
    删除向量库中指定文档的所有信息
    :param pdf_filename: 要删除的文档
    :return: 操作结果提示
    """
    if not isinstance(pdf_filename, str) or len(pdf_filename.strip()) == 0:
        return "错误：输入的文件名不能为空，且必须是字符串类型"
    # 1. 检查向量库是否存在
    if not os.path.exists(config.PERSIST_DIR):
        return "本地向量库不存在，无需删除"

    # 2. 加载向量库
    try:
        embeddings = get_embeddings()
        config.vectordb = FAISS.load_local(config.PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        error_msg=traceback.format_exc()[:200]
        return f"错误：向量库调用失败\n{error_msg}"
    # 3. 筛选要删除的文档（根据source元数据）
    delete_ids = []
    for idx, doc in config.vectordb.docstore._dict.items():
        try:
            # doc_content是完整Document对象（有metadata）
            if hasattr(doc, 'metadata'):
                doc_source = doc.metadata.get("source", "").lower()
            else:
                # 跳过无metadata的纯字符串（避免报错）
                continue
        except:
            continue
        if doc_source == pdf_filename.lower():
            delete_ids.append(idx)  # 原生doc_id，FAISS能识别
    # 4. 执行删除
    if not delete_ids:
        return f"未找到相关文档「{pdf_filename}」的相关数据"

    config.vectordb.delete(delete_ids)  # 核心：删除指定ID的向量
    config.vectordb.save_local(config.PERSIST_DIR)  # 重新保存向量库（覆盖原有文件）
    print(f"已删除PDF「{pdf_filename}」的{len(delete_ids)}个文本块")
    return f"成功删除PDF「{pdf_filename}」的{len(delete_ids)}个文本块，向量库已更新"

if __name__ == '__main__':
    vector=load_exist_vector_db("./faiss_db_cpu")

    context_with_source, final_docs=enhanced_retrieval(vector,"LLM是什么？")
    print(context_with_source)
    print("*"*80)
    print(final_docs)
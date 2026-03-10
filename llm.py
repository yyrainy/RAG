
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import config
import tools
import gradio as gr

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_community.llms import BaseLLM
from langchain_core.outputs import GenerationChunk,LLMResult,Generation
from typing import Optional,Any,List,Iterator
import requests
import json
#这好像是langchain和modelscope之间数据适配的问题所以才有的这个，直接不用管就行
class ModelScopeAPILLM(BaseLLM):
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    api_base: str = "https://api-inference.modelscope.cn/v1"

    @property
    def _llm_type(self) -> str:
        return "modelscope_api"

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        prompt = prompts[0]
        messages = [{"role": "user", "content": prompt}]
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"Error: {str(e)}"
        generations = [[Generation(text=text)]]
        return LLMResult(generations=generations)

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        messages = [{"role": "user", "content": prompt}]
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                data_str = line[6:]  # 去掉 "data: " 前缀
                if data_str == '[DONE]':
                    break
                reasoning,content="",""
                try:
                    data = json.loads(data_str)
                    # 提取内容，兼容不同格式
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if "delta" in choice:
                            reasoning=choice["delta"].get("reasoning_content","")
                            content = choice["delta"].get("content", "")
                        elif "message" in choice:
                            content = choice["message"].get("content", "")
                        else:
                            content = ""
                    if reasoning or content:
                        # 将思考部分用灰色 span 包裹，回答部分正常
                        html = ""
                        if reasoning:
                            reasoning=reasoning.replace("\n","<br>")
                            html += f'<span style="color:gray;">{reasoning}</span>'
                        if content:
                            content = content.replace("\n", "<br>")
                            if reasoning:
                                html += "<br>"
                            html += content
                        yield GenerationChunk(text=html)
                except json.JSONDecodeError:
                    continue

##########################################################################


def get_session_history(session_id: str):
    """获取/创建指定ID的会话历史"""
    senssion_data=config.SESSION_STORAGE[session_id]
    if not isinstance(senssion_data,ChatMessageHistory):
        config.SESSION_STORAGE[session_id]["chat_history"]=ChatMessageHistory()

    # print(f"会话历史返回值类型：{type(config.SESSION_STORAGE[session_id]['chat_history'])}")
    return config.SESSION_STORAGE[session_id]["chat_history"]


def get_radio_update(session_id):
    choices = [(config.SESSION_STORAGE[sid]["name"], sid) for sid, data in config.SESSION_STORAGE.items() if
               data.get("visible", False)]
    if any(session_id==sid for _,sid in choices):
        return gr.update(choices=choices,value=session_id)
    else:
        return gr.update(choices=choices, value=None)

# 构建模型（调用API做下适配）
def build_rag_chain():
    """
    构建RAG链，支持多会话切换
    """
    # 幻觉抑制Prompt（保留）
    prompt_with_history = ChatPromptTemplate.from_template("""
    你是一个严谨且聪明的知识库问答助手，必须严格遵守以下规则：
    1. 仅使用【参考信息】中的内容回答问题，绝不编造任何未提及的信息；
    2. 如果【参考信息】中没有相关内容，就用你自己的理解来回答；
    3. 回答要简洁、准确，最后必须标注信息来源（参考信息1/2/3）；


    【历史对话】
    {chat_history}

    【参考信息】
    {context}

    【用户问题】
    {question}
    """)

    def rag_answer(question, vectordb, chat_history, session_id):
        """新增session_id参数，支持多会话"""
        if vectordb is None:
            chat_history.append({"role": "assistant", "content": "向量库为空！"})
            yield gr.update(value=""), gr.update(value=chat_history), session_id
            return
        if not question or question.strip() == "":
            chat_history.append({"role": "assistant", "content": "请输入有效的问题！"})
            yield gr.update(value=""), gr.update(value=chat_history), session_id
            return

        # 添加用户问题到前端聊天记录
        chat_history.append({"role": "user", "content": question.strip()})

        # 检索相关文档
        context, _ = tools.enhanced_retrieval(vectordb, question.strip())
        #下面用到的变量尽量放在这里
        stream_sucess = False
        update = False
        init_radio = get_radio_update(session_id)
        #for循环模型调用流程
        # 遍历模型列表，出错切换
        for idx, candidate in enumerate(config.LLM_CANDIDATES):
            model_id = candidate["model_id"]
            model_name = candidate["name"]
            try:
                print(f"尝试调用模型：{model_name}")
                llm = ModelScopeAPILLM(
                    model=model_id,
                    temperature=config.temperature,
                    max_tokens=config.max_token,
                    api_key=config.api_key,
                )
                # 构建基础RAG链
                base_rag_chain = (
                        RunnablePassthrough.assign(
                            context=lambda x: x["context"]
                        )
                        | prompt_with_history
                        | llm
                        | StrOutputParser()
                )
                # 包装成带会话历史的链
                rag_chain_with_history = RunnableWithMessageHistory(
                    base_rag_chain,
                    get_session_history,  # 绑定当前会话ID
                    input_messages_key="question",      #下面3个都是类似起个名字例如：名字是question那么输入参数必须是question字典
                    history_messages_key="chat_history",
                    output_messages_key="output"
                )

                # 流式调用
                stream = rag_chain_with_history.stream(
                    {"question": question.strip(),"context":context},
                    config={"configurable": {"session_id": session_id}} #RunnableWithMessageHistory规定 configurable中找参数
                )
                #这个前端打印不是一次性，是为了流式输出每for一次输出一次
                temp_answer = ""
                for trunk in stream:
                    if trunk:
                        temp_answer += trunk
                        # 更新前端聊天记录
                        if len(chat_history) > 0 and chat_history[-1]["role"] == "assistant":
                            chat_history[-1]["content"] = temp_answer
                        else:
                            chat_history.append({"role": "assistant", "content": temp_answer})
                        yield gr.update(value=""), gr.update(value=chat_history), session_id,init_radio

                stream_sucess = True
                # 自动更新会话名称（取前10个字符）
                # print(config.SESSION_STORAGE[session_id]["name"])
                # print(session_id)
                # print(len(config.SESSION_STORAGE))
                if config.SESSION_STORAGE[session_id]["name"] in ["新会话", f"新会话{len(config.SESSION_STORAGE)-1}"]:
                    config.SESSION_STORAGE[session_id]["name"] = question.strip()[:15] + "..."
                # print(config.SESSION_STORAGE[session_id]["name"])
                break

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "速率限制" in error_msg or "HTTP" in error_msg or "RequestException" in error_msg:
                    print(f"模型 {model_name} 调用失败（原因：{error_msg[:50]}），切换下一个模型...")
                    if idx == len(config.LLM_CANDIDATES) - 1:
                        error_msg = f"所有模型均调用失败（限流/连接错误），参考信息：\n{context}"
                        chat_history.append({"role": "assistant", "content": error_msg})
                        # yield gr.update(value=""), gr.update(value=chat_history), session_id,init_radio
                    continue
                else:
                    print(f"模型 {model_name} 调用失败（非限流：{error_msg[:50]}），切换下一个模型...")
                    if idx == len(config.LLM_CANDIDATES) - 1:
                        chat_history.append({"role": "assistant", "content": error_msg})
                        # yield gr.update(value=""), gr.update(value=chat_history), session_id,init_radio
                    continue

        if not stream_sucess:
            answer = f"所有模型均调用失败，参考信息：\n{context}"
            chat_history.append({"role": "assistant", "content": answer})
            # yield gr.update(value=""), gr.update(value=chat_history), session_id,get_radio_update(session_id)
        if not config.SESSION_STORAGE[session_id]["visible"]:
            config.SESSION_STORAGE[session_id]["visible"] = True
            update=True
        final_radio_update=get_radio_update(session_id) if update else init_radio
        # print("LLM查询的结果")
        # print(final_radio_update)
        yield gr.update(value=""), gr.update(value=chat_history), session_id, final_radio_update
    return rag_answer

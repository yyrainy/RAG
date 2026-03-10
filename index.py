
import os
import tools
import gradio as gr
import shutil
import llm
import config
import uuid
from langchain_community.chat_message_histories import ChatMessageHistory

# 工具函数：获取会话列表（用于前端展示）
def get_session_list():
    """返回所有会话的名称列表（ID|名称）"""
    session_list=[]
    for sid,data in config.SESSION_STORAGE.items():
        if data.get("visible",False):
            session_list.append((data["name"],sid))
    # print(session_list)
    return session_list
    # return [(config.SESSION_STORAGE[sid]["name"],sid) for sid,data in config.SESSION_STORAGE.items() if data.get("visible",False)]


# 工具函数：切换会话
def switch_session(selected_session,current_session):
    """点击历史会话时，恢复对应的聊天记录"""
    # print("开始切换")
    # print(selected_session)
    if selected_session is None or selected_session not in config.SESSION_STORAGE:
        session_id = current_session
    else:
        session_id = selected_session
    # print(session_id)
    # 解析会话ID
    # 从会话存储中获取历史记录
    session_chat_history = config.SESSION_STORAGE[session_id]["chat_history"]
    # print(type(session_chat_history))
    # 转换为Gradio Chatbot格式
    gradio_chat_history = []
    # print("切换结束")
    for msg in session_chat_history.messages:
        if msg.type == "human":
            role = "user"
        else: role = "assistant"
        gradio_chat_history.append({"role": role, "content": msg.content})
    return gradio_chat_history, session_id

# 工具函数：新建会话
def create_new_session(current_sension_id):
    """创建新会话，返回新会话ID和更新后的会话列表"""
    # print("开始创建")
    # print(current_sension_id)
    new_session_list = get_session_list()
    if current_sension_id in config.SESSION_STORAGE:
        current_sension=config.SESSION_STORAGE[current_sension_id]
        if not current_sension["visible"] and len(current_sension["chat_history"].messages) == 0:
            # print("复用")
            return gr.update(choices=new_session_list, value=None), [], current_sension_id

    new_session_id = f"session_{uuid.uuid4().hex[:8]}"
    config.SESSION_STORAGE[new_session_id] = {
        "name": f"新会话{len(config.SESSION_STORAGE)}",
        "chat_history": ChatMessageHistory(),
        "visible": False,
    }
    # print("创建结束")
    return gr.update(choices=new_session_list, value=None), [], new_session_id


# 工具函数：删除会话
def delete_session(selected_session):
    """删除选中的会话"""
    # print("开始删除")
    # print(selected_session)
    if selected_session in config.SESSION_STORAGE:
        del config.SESSION_STORAGE[selected_session]
        new_options = get_session_list()
        if new_options:
            new_current=new_options[0][1]
            return (gr.update(choices=new_options, value=new_current),
                    [],new_current)
        else:
            new_session_id = f"session_{uuid.uuid4().hex[:8]}"
            config.SESSION_STORAGE[new_session_id] = {
                "name": f"新会话{len(config.SESSION_STORAGE)}",
                "chat_history": ChatMessageHistory(),
                "visible": False,
            }
            return (gr.update(choices=new_options, value=None),
                    [],new_session_id)



# 原有工具函数：获取已加载文档
def get_loaded_files():
    if config.vectordb is None:
        return "当前无已加载的文档"
    try:
        file_list = set()
        for _, doc in config.vectordb.docstore._dict.items():
            if hasattr(doc, 'metadata'):
                file_list.add(doc.metadata.get("source", "未知文档").strip())
        if file_list:
            return "当前已加载的文档：\n" + "\n".join([f"{file}" for file in file_list])
        else:
            return "向量库为空"
    except Exception as e:
        print(f"获取文档列表失败：{e}")
        return "无法获取已加载的文档列表"


# 上传文档构建向量库
def upload_file_and_build_db(files):
    if files is None or len(files) == 0:
        return "请上传至少一个文档（支持PDF/Word/TXT）！", get_loaded_files()
    SUPPORTED_FORMATS = (".pdf", ".docx", ".txt")
    invalid_files = []
    valid_files = []
    for file_path in files:
        if not file_path.lower().endswith(SUPPORTED_FORMATS):
            invalid_files.append(os.path.basename(file_path))
        else:
            valid_files.append(file_path)

    if invalid_files:
        return f"以下文件格式不支持：{', '.join(invalid_files)}\n仅支持PDF/Word/TXT！", get_loaded_files()

    all_doc_splits = []
    upload_status = []
    try:
        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                doc_splits = tools.load_and_split_pdf(file_path)
                all_doc_splits.extend(doc_splits)
                upload_status.append(f"{file_name}：生成{len(doc_splits)}个文本块")
            except Exception as e:
                upload_status.append(f"{file_name}：处理失败 - {str(e)}")

        if all_doc_splits:
            config.vectordb = tools.build_vector_db(all_doc_splits)
            status_text = "\n".join(upload_status)
            return f"批量上传处理完成！\n{status_text}\n\n总计生成{len(all_doc_splits)}个文本块\n", get_loaded_files()
        else:
            return f"所有文件处理失败：\n{chr(10).join(upload_status)}", get_loaded_files()
    except Exception as e:
        return f"批量上传失败：{str(e)}", get_loaded_files()


# 适配gradio前端显示
def upload_handler(files):
    op_status, _ = upload_file_and_build_db(files)
    status_text = get_loaded_files()
    return [op_status] + [status_text] * len(config.db_status_components)


# 删除文档数据
def delete_handler(filename):
    if not filename:
        op_status = "请输入要删除的文档文件名！"
    else:
        op_status = tools.delete_file_from_vector_db(filename)
    config.vectordb = tools.load_exist_vector_db()
    status_text = get_loaded_files()
    return [op_status] + [status_text] * len(config.db_status_components)


# 清空所有文档
def clear_all_file():
    if os.path.exists(config.PERSIST_DIR):
        try:
            shutil.rmtree(config.PERSIST_DIR)
            op_status = "已清空所有文档数据（磁盘+内存）"
        except Exception as e:
            op_status = f"清空失败：{str(e)}"
    else:
        op_status = "当前无向量库数据，无需清空"
    config.vectordb = None
    status_text = get_loaded_files()
    return [op_status] + [status_text] * len(config.db_status_components)


# 调用LLM回答问题
def answer_question(question, chat_history, session_id):
    rag_answer = llm.build_rag_chain()
    for result in rag_answer(question, config.vectordb, chat_history, session_id):
        yield result


# 清空当前会话的聊天记录
def clear_chat(session_id):
    if session_id in config.SESSION_STORAGE:
        config.SESSION_STORAGE[session_id]["chat_history"] =ChatMessageHistory()
    return gr.update(value=""), gr.update(value=[])

#不用管只是修改了一下前端显示
css="""          
    /* 针对 session-radio 的每个选项进行样式设置 */
    #session-radio .wrap label{
        white-space: nowrap;        /* 禁止换行 */
        height: 30px;               /* 固定高度 */
        width: 500px;
        line-height: 30px;          /* 垂直居中 */
        margin: 2px 0;              /* 可选间距 */
    }

"""
# 主界面构建
def gr_show():                                      #css这个不用管只是调了一下样式
    with gr.Blocks(title="RAG知识库问答系统",) as demo:
        gr.Markdown("# RAG本地知识库问答系统")
        gr.Markdown("---")
        with gr.Tab("### 文档管理"):
            # 文档上传Tab
            with gr.Tab("文档上传"):
                files = gr.File(label="上传文档（PDF/Word/TXT）", type="filepath", file_count="multiple")
                build_btn = gr.Button("构建向量库（纯CPU）")
                build_status = gr.Textbox(label="构建状态", lines=3)
                db_status1 = gr.Textbox(
                    label="已加载文档",
                    value=get_loaded_files(),
                    lines=5,
                    interactive=False
                )
                config.db_status_components.append(db_status1)

            # 文档删除Tab
            with gr.Tab("删除文档"):
                filename = gr.Textbox(label="文档名", placeholder="例如：数据结构.pdf")
                delete_doc_btn = gr.Button("删除该文档", variant="primary")
                clear_doc_btn = gr.Button("清空所有文档", variant="secondary")
                delete_status = gr.Textbox(label="删除结果", lines=3)
                db_status2 = gr.Textbox(
                    label="已加载文档",
                    value=get_loaded_files(),
                    lines=5,
                    interactive=False
                )
                config.db_status_components.append(db_status2)
        # 全局状态：当前选中的会话ID
        current_session_id = gr.State(value=config.DEFAULT_SESSION_ID)
        with gr.Tab("### 智能问答"):
            # 布局：左侧（会话管理+文档管理），右侧（聊天区）
            with gr.Row():
                # 左侧面板：会话管理 + 文档管理
                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("### 会话管理")
                    gr.Markdown("---")
                    with gr.Row():
                        new_session_btn = gr.Button("新建会话", variant="primary")
                        delete_session_btn = gr.Button("删除会话", variant="secondary")
                    session_status = gr.Textbox(label="会话操作状态", lines=1)

                    session_radio = gr.Radio(
                        choices=get_session_list(),
                        label="历史会话",
                        value=config.DEFAULT_SESSION_ID,
                        interactive=True,
                        container=True, # 保持容器样式
                        elem_id = "session-radio"
                    )
                # 右侧面板：聊天区
                with gr.Column(scale=4):
                    # 聊天记录
                    chatbot = gr.Chatbot(
                        label="对话历史（流式输出）",
                        height=600,
                        resizable=True,
                        avatar_images=(None, "https://img.icons8.com/fluency/96/000000/robot.png"),
                        sanitize_html = False,
                    )
                    # 输入框 + 按钮
                    with gr.Row():
                        question = gr.Textbox(
                            label="输入问题",
                            placeholder="例如：论文中提出的核心方法是什么？",
                            scale=8
                        )
                        answer_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_chat_btn = gr.Button("清空当前会话", variant="secondary")

        # 绑定事件：会话管理
        # 切换会话
        session_radio.change(
            fn=switch_session,
            inputs=[session_radio,current_session_id],
            outputs=[chatbot, current_session_id]
        )

        # 新建会话
        new_session_btn.click(
            fn=create_new_session,
            inputs=[current_session_id],
            outputs=[session_radio, chatbot,current_session_id],
            # api_name="create_session"
        ).then(
            fn=lambda: "新建会话成功",
            outputs=[session_status]
        )

        # 删除会话
        delete_session_btn.click(
            fn=delete_session,
            inputs=[current_session_id],
            outputs=[session_radio,chatbot,current_session_id]
        ).then(
            fn=lambda :"已删除会话",
            outputs=[session_status]
        )

        # 绑定事件：文档管理
        build_btn.click(upload_handler, inputs=files, outputs=[build_status] + config.db_status_components)
        delete_doc_btn.click(delete_handler, inputs=filename, outputs=[delete_status] + config.db_status_components)
        clear_doc_btn.click(clear_all_file, outputs=[delete_status] + config.db_status_components)

        # 绑定事件：聊天交互  answer_btn和question一样一个是按钮触发一个是回车触发
        question.submit(
            fn=answer_question,
            inputs=[question, chatbot, current_session_id],
            outputs=[question, chatbot, current_session_id,session_radio]
        )
        answer_btn.click(
            fn=answer_question,
            inputs=[question, chatbot, current_session_id],
            outputs=[question, chatbot, current_session_id,session_radio]
        )
        clear_chat_btn.click(
            fn=clear_chat,
            inputs=[current_session_id],
            outputs=[question, chatbot]
        )

    # 启动界面
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,css=css)
#

import gradio as gr
from src.ui import chat_with_llm

with gr.Blocks() as demo:
    gr.Markdown("# YouTube Transcript RAG Chatbot")
    gr.Markdown("Paste YouTube Video URL or Video Id. The AI will respond and remember the conversation context.")

    # A State component to store the actual Langchain chat history objects
    chat_history = gr.State({"current_video_id": None, "ingestion_done": False, "history": []})

    with gr.Row():
        with gr.Column(scale=1):
            video_id_input = gr.Textbox(lines=2, label="YouTube video ID or URL:", placeholder="Paste YouTube URl or Video Id...", elem_id="videoId_input_box")
        with gr.Column(scale=2):
            user_input = gr.Textbox(lines=3, label="Your Query:", placeholder="Ask your query (type 'exist' to stop)...", elem_id="user_input_box")
            submit_btn = gr.Button("Send")
        with gr.Column(scale=3):
            bot_output = gr.Textbox(lines=10, label="Bot Response:", interactive=False, elem_id="ai_output_box")

    full_history_display = gr.Textbox(lines=15, label="Conversation History:", interactive=False, elem_id="history_display_box")

    # When user submits input or clicks send button
    submit_event = submit_btn.click(
        fn=lambda: (
            gr.update(interactive=False),
            gr.update(interactive=False),
        ),
        inputs=None,
        outputs=[submit_btn, user_input]
    ).then(
        fn=chat_with_llm,
        inputs=[video_id_input,user_input,chat_history],
        outputs=[bot_output, chat_history,full_history_display,video_id_input],
        queue=False
    ).then(
        fn=lambda: gr.update(value=""),
        inputs=None,
        outputs=[user_input]
    ).then(
        fn=lambda: (
            gr.update(interactive=True),
            gr.update(interactive=True),
        ),
        inputs=None,
        outputs=[submit_btn, user_input]
    )

    video_id_input.submit(
        fn=chat_with_llm,
        inputs=[video_id_input,user_input,chat_history],
        outputs=[bot_output,chat_history,full_history_display,video_id_input],
        queue=False
        ).then(
            lambda: gr.update(value=""),
            inputs=None,
            outputs=[user_input]
        )

    user_input.submit(
        fn=lambda: (
            gr.update(interactive=False),
            gr.update(interactive=False)
        ),
        inputs=None,
        outputs=[submit_btn, user_input]
    ).then(
        fn=chat_with_llm,
        inputs=[video_id_input,user_input, chat_history],
        outputs=[bot_output, chat_history, full_history_display],
        queue=False
    ).then(
        lambda: gr.update(value=""),
        inputs=None,
        outputs=[user_input]
    ).then(
        fn=lambda: (
        gr.update(interactive=True),
        gr.update(interactive=True)
    ),
    inputs=None,
    outputs=[submit_btn, user_input]
    )

    def clear_all():
        return (
            gr.update(value="",interactive=True),                               #video_id_input
            "",                                                                 #user_input
            {"current_video_id": None, "ingestion_done": False, "history": []}, #chat_history
            "",                                                                 #bot_output
            "",                                                                 #full_history_display
        )

    gr.Button("Clear").click(
        clear_all,
        inputs=[], 
        outputs=[video_id_input,user_input,chat_history,bot_output,full_history_display]
    )

demo.launch(debug=True) #share=True
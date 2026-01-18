import gradio as gr
from urllib.parse import urlparse, parse_qs
from src.services import get_transcript,user_query_response
from langchain_core.messages import HumanMessage,AIMessage


#Function to get video if from youtube url
def from_url_get_video_id(url_or_id):
    if url_or_id.startswith("http"):
        parsed_url = urlparse(url_or_id)
        video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        return video_id
    return url_or_id

#Funtion for chat history
def format_history(history_list):
    output_string = ""
    for msg in history_list:
        if isinstance(msg,HumanMessage):
            output_string +=f"User: {msg.content}\n"
        elif isinstance(msg,AIMessage):
            output_string +=f"Bot: {msg.content}\n"
    return output_string


def chat_with_llm(video_id,user_query,state):
    current_video_id = state.get("current_video_id")
    ingestion_done = state.get("ingestion_done", False)
    history = state.get("history", [])

    # STOP Condition
    if user_query.lower().strip() in ["stop", "exist", "exists", "quit"]:
        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content="Thanks for the chat!"))
        state["history"] = history
        new_state = {"current_video_id": None, "ingestion_done": False, "history": history}
        return (
            "Chat ended",
            new_state,
            format_history(history),
            gr.update(value="",interactive=True)
        )

    # If no ingestion done yet → treat as first message
    if not ingestion_done:
        if not video_id:
            return ("Please enter a YouTube URL or ID.", state, format_history(history), gr.update())
        if not user_query or user_query.strip() == "":
            return ("Query cannot be empty!.", state, format_history(history), gr.update())
        
        current_video_id=from_url_get_video_id(video_id)
        # Ingest transcript
        transcript = get_transcript(current_video_id)
        if not transcript:
            history.append(AIMessage(content="Transcript failed to load."))
            return ("Transcript ingestion failed.", state, format_history(history),gr.update())
        
        ingestion_done = True

        # Add the current human message to the internal chat history
        history.append(HumanMessage(content=user_query))

        # Retrieval
        query_response = user_query_response(user_query,current_video_id)
        if query_response["result"] == "No data found":
            final_answer = "The answer is not available in the provided video transcript."
        else:
            final_answer = query_response["result"]


        if len(query_response.get('pages',[]))>0 :
            final_answer += f"\n\nSource pages: {query_response['pages']}"

        # Add the AI's response to the internal chat history
        history.append(AIMessage(content=final_answer))

        # Update state
        new_state = {
            "current_video_id": current_video_id,
            "ingestion_done": True,
            "history": history
        }

        # Return AI response, the updated internal_history_state, and the display_history
        return final_answer,new_state,format_history(history),gr.update(interactive=False)
    
    # Subsequent messages (NO INGESTION)
    if not user_query or user_query.strip() == "":
            return ("Query cannot be empty!.", state, format_history(history), gr.update())
    history.append(HumanMessage(content=user_query))
    query_response = user_query_response(user_query, current_video_id)
    if query_response["result"] == "No data found":
        final_answer = "The answer is not available in the provided video transcript."
    else:
        final_answer = query_response["result"]

    if len(query_response.get("pages", [])) > 0:
        final_answer += f"\nPages: {query_response['pages']}"

    history.append(AIMessage(content=final_answer))

    # Save
    state["history"] = history

    return (
        final_answer,
        state,
        format_history(history),
        gr.update(interactive=False)
    )


# Gradio Interface using gr.Blocks for more control
# with gr.Blocks() as demo:
#     gr.Markdown("# YouTube Transcript RAG Chatbot")
#     gr.Markdown("Paste YouTube Video URL or Video Id. The AI will respond and remember the conversation context.")

#     # A State component to store the actual Langchain chat history objects
#     chat_history = gr.State({"current_video_id": None, "ingestion_done": False, "history": []})

#     with gr.Row():
#         with gr.Column(scale=1):
#             video_id_input = gr.Textbox(lines=2, label="YouTube video ID or URL:", placeholder="Paste YouTube URl or Video Id...", elem_id="videoId_input_box")
#         with gr.Column(scale=2):
#             user_input = gr.Textbox(lines=3, label="Your Query:", placeholder="Ask your query (type 'exist' to stop)...", elem_id="user_input_box")
#             submit_btn = gr.Button("Send")
#         with gr.Column(scale=3):
#             bot_output = gr.Textbox(lines=10, label="Bot Response:", interactive=False, elem_id="ai_output_box")

#     full_history_display = gr.Textbox(lines=15, label="Conversation History:", interactive=False, elem_id="history_display_box")

#     # When user submits input or clicks send button
#     submit_event = submit_btn.click(
#         fn=lambda: (
#             gr.update(interactive=False),
#             gr.update(interactive=False),
#         ),
#         inputs=None,
#         outputs=[submit_btn, user_input]
#     ).then(
#         fn=chat_with_llm,
#         inputs=[video_id_input,user_input,chat_history],
#         outputs=[bot_output, chat_history,full_history_display,video_id_input],
#         queue=False
#     ).then(
#         fn=lambda: gr.update(value=""),
#         inputs=None,
#         outputs=[user_input]
#     ).then(
#         fn=lambda: (
#             gr.update(interactive=True),
#             gr.update(interactive=True),
#         ),
#         inputs=None,
#         outputs=[submit_btn, user_input]
#     )

#     video_id_input.submit(
#         fn=chat_with_llm,
#         inputs=[video_id_input,user_input,chat_history],
#         outputs=[bot_output,chat_history,full_history_display,video_id_input],
#         queue=False
#         ).then(
#             lambda: gr.update(value=""),
#             inputs=None,
#             outputs=[user_input]
#         )

#     user_input.submit(
#         fn=lambda: (
#             gr.update(interactive=False),
#             gr.update(interactive=False)
#         ),
#         inputs=None,
#         outputs=[submit_btn, user_input]
#     ).then(
#         fn=chat_with_llm,
#         inputs=[video_id_input,user_input, chat_history],
#         outputs=[bot_output, chat_history, full_history_display],
#         queue=False
#     ).then(
#         lambda: gr.update(value=""),
#         inputs=None,
#         outputs=[user_input]
#     ).then(
#         fn=lambda: (
#         gr.update(interactive=True),
#         gr.update(interactive=True)
#     ),
#     inputs=None,
#     outputs=[submit_btn, user_input]
#     )

#     def clear_all():
#         return (
#             gr.update(value="",interactive=True),                               #video_id_input
#             "",                                                                 #user_input
#             {"current_video_id": None, "ingestion_done": False, "history": []}, #chat_history
#             "",                                                                 #bot_output
#             "",                                                                 #full_history_display
#         )

#     gr.Button("Clear").click(
#         clear_all,
#         inputs=[], 
#         outputs=[video_id_input,user_input,chat_history,bot_output,full_history_display]
#     )

# demo.launch(debug=True) #share=True
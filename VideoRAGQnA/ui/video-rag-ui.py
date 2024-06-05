import os
import time
import threading
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, set_seed
import streamlit as st

import requests

from utils import config_reader as reader
from utils import prompt_handler as ph

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
vectordb_service_host_ip = os.getenv("VECTORDB_SERVICE_HOST_IP")

set_seed(22)

if 'config' not in st.session_state.keys():
    st.session_state.config = reader.read_config('/ui/config.yaml')

config = st.session_state.config

vectordb_service_host_port = int(config['vector_db']['port'])
model_path = config['model_path']
video_dir = config['videos']
print(video_dir)
video_dir = video_dir.replace('../', '')
print(video_dir)
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

st.title("Video RAG")

title_alignment="""
<style>
h1 {
  text-align: center
}

video.stVideo {
    width: 200px;
    height: 500px;      
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

def set_proxy(addr:str):
    # for DNS: "http://child-prc.intel.com:913"
    # for Huggingface downloading: "http://proxy-igk.intel.com:912"
    os.environ['http_proxy'] = addr
    os.environ['https_proxy'] = addr
    os.environ['HTTP_PROXY'] = addr
    os.environ['HTTPS_PROXY'] = addr

@st.cache_resource       
def load_models():
    print("HF Token: ", HUGGINGFACEHUB_API_TOKEN)
    # set_proxy("http://proxy-igk.intel.com:912") # specific for PRC Usage
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map='auto', trust_remote_code=True, token=HUGGINGFACEHUB_API_TOKEN
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=HUGGINGFACEHUB_API_TOKEN)
    tokenizer.padding_size = 'right'
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    # set_proxy("http://child-prc.intel.com:913") # specific for PRC Usage
    return model, tokenizer, streamer

model, tokenizer, streamer = load_models()

class CustomLLM(LLM):
        
    @torch.inference_mode()
    def _call(
            self, 
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            streamer: Optional[TextIteratorStreamer] = None,  # Add streamer as an argument
        ) -> str:
        
        tokens = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            output = model.generate(input_ids = tokens,
                                    max_new_tokens = 100,
                                    num_return_sequences = 1,
                                    num_beams = 1,
                                    min_length = 1,
                                    top_p = 0.9,
                                    top_k = 50,
                                    repetition_penalty = 1.2,
                                    length_penalty = 1,
                                    temperature = 0.1,
                                    streamer=streamer,
                                    # pad_token_id=tokenizer.eos_token_id,
                                    do_sample=True
                    )
        
    def stream_res(self, prompt):
        thread = threading.Thread(target=self._call, args=(prompt, None, None, streamer))  # Pass streamer to _call
        thread.start()
        
        for text in streamer:
            yield text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return model_path # {"name_of_model": model_path}

    @property
    def _llm_type(self) -> str:
        return "custom"
    
def get_top_doc(results, qcnt):
    hit_score = {}
    if results == None:
        return None
    for r in results:
        try:
            video_name = r['metadata']['video']
            if video_name not in hit_score.keys(): hit_score[video_name] = 0
            hit_score[video_name] += 1
        except KeyError as r:
            print("no video name", r)

    x = dict(sorted(hit_score.items(), key=lambda item: -item[1]))
    
    if qcnt >= len(x):
        return None
    print (f'top docs = {x}')
    return {'video': list(x)[qcnt]}

def play_video(x):
    if x is not None:
        video_file = x.replace('.pt', '')
        path = video_dir + video_file
        
        video_file = open(path, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes, start_time=0)

def get_data(api_url:str, query:dict):
    try:
        response = requests.get(api_url, query)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None

if 'llm' not in st.session_state.keys():
    with st.spinner('Loading Models . . .'):
        time.sleep(1)
        st.session_state['llm'] = CustomLLM()
        
if 'vs' not in st.session_state.keys():
    with st.spinner('Preparing RAG pipeline'):
        time.sleep(1)
        api_url = f"http://{vectordb_service_host_ip}:{vectordb_service_host_port}/health"
        response = get_data(api_url, {})
        st.session_state['vs'] = "done"
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
def clear_chat_history():
    st.session_state.example_video = 'Enter Text'
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
def RAG(prompt):
    
    with st.status("Querying database . . . ", expanded=True) as status:
        st.write('Retrieving 3 image docs') #1 text doc and 
        api_url = f"http://{vectordb_service_host_ip}:{vectordb_service_host_port}/visual_rag_retriever/query"
        results = get_data(api_url, {"prompt": prompt})
        status.update(label="Retrived Top matching video!", state="complete", expanded=False)
    
    print (f'prompt={prompt}\n')
                
    top_doc = get_top_doc(results, st.session_state['qcnt'])
    print ('TOP DOC = ', top_doc)
    if top_doc == None:
        return None, None
    video_name = top_doc['video']
    
    return video_name, top_doc

def get_description(vn):
    content = None
    des_path = os.path.join(config['description'], vn + '.txt')
    with open(des_path, 'r') as file:
        content = file.read()
    return content
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if 'prevprompt' not in st.session_state.keys():
    st.session_state['prevprompt'] = ''
    print("Setting prevprompt to None")
if 'prompt' not in st.session_state.keys():
    st.session_state['prompt'] = ''
if 'qcnt' not in st.session_state.keys():
    st.session_state['qcnt'] = 0

def handle_message():
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        # Handle user messages here
        with st.chat_message("assistant"):
            placeholder = st.empty()
            start = time.time()
            prompt = st.session_state['prompt']
            
            if prompt == 'Find similar videos':
                prompt = st.session_state['prevprompt']
                st.session_state['qcnt'] += 1
            else:
                st.session_state['qcnt'] = 0
                st.session_state['prevprompt'] = prompt
            video_name, top_doc = RAG(prompt)
            if video_name == None:
                full_response = f"No more relevant videos found. Select a different query. \n\n"
                placeholder.markdown(full_response)
                end = time.time()
            else:
                with col2:
                    play_video(video_name)
                
                scene_des = get_description(video_name)
                formatted_prompt = ph.get_formatted_prompt(scene=scene_des, prompt=prompt)
                print(f'formatted_prompt = {formatted_prompt}')
                full_response = ''
                full_response = f"Most relevant retrived video is **{video_name}** \n\n"
                
                for new_text in st.session_state.llm.stream_res(formatted_prompt):
                    print(f'new_text = {new_text}')
                    full_response += new_text
                    placeholder.markdown(full_response)
                
                end = time.time()
                full_response += f'\n\n🚀 Generated in {end - start} seconds.'
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
      
def display_messages():
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

col1, col2 = st.columns([2, 1])

with col1:
    st.selectbox(
        'Example Prompts',
        (
            'Enter Text', 
            'Find similar videos', 
            'Man wearing glasses', 
            'People reading item description',
            'Man holding red shopping basket',
            'Was there any person wearing a blue shirt seen today?',
            'Was there any person wearing a blue shirt seen in the last 6 hours?',
            'Was there any person wearing a blue shirt seen last Sunday?',
            'Was a person wearing glasses seen in the last 30 minutes?',
            'Was a person wearing glasses seen in the last 72 hours?',
        ),
        key='example_video'
    )

    st.write('You selected:', st.session_state.example_video)

if st.session_state.example_video == 'Enter Text':
    if prompt := st.chat_input(disabled=False):
        st.session_state['prompt'] = prompt
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        if prompt == 'Find similar videos':            
            st.session_state.messages.append({"role": "assistant", "content": "Not supported"})
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
else:
    prompt = st.session_state.example_video
    st.session_state['prompt'] = prompt
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.chat_input(disabled=True)
    if prompt == 'Find similar videos':
        st.session_state.messages.append({"role": "user", "content": prompt+': '+st.session_state['prevprompt']})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

with col1:
    display_messages()
    handle_message()

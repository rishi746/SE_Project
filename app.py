"""Local CPU document workspace with explicit demo; model execution stays on Kaggle."""
from pathlib import Path
import sys
import tempfile
import json
import os
import streamlit as st

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from sms_ac.documents import read_design_docx, DocumentError
from sms_ac.pipeline import SMSACPipeline, DemoBackend, DEMO_REQUEST, KaggleAdapterBackend, InferenceError

st.set_page_config(page_title='SMS AC · Design and lifecycle',page_icon='📄',layout='wide')
st.title('SMS AC')
st.write('Network application design → Word report → explained lifecycle recommendation')
ON_KAGGLE = os.name != 'nt' and Path('/kaggle/working').is_dir()
st.info('Kaggle mode: trained adapter manifests and weights are checked before model use.' if ON_KAGGLE else
        'Local CPU workspace: collect project notes, inspect documents and explore the labeled demo. Actual model replies run on Kaggle after training.')

def trained_pipeline():
    if 'planning_backend' not in st.session_state:
        st.session_state['planning_backend'] = KaggleAdapterBackend(ROOT/'models')
    return SMSACPipeline(st.session_state['planning_backend'],output_dir=ROOT/'outputs/planning')
planning, demo, document, training = st.tabs(['Discuss your project','Explore the workflow','Read a design document','Kaggle training'])
with planning:
    st.subheader('Planning and requirements discussion')
    on_kaggle = ON_KAGGLE
    st.caption('The design specialist asks follow-up questions before creating the report. Unknown details remain explicit.')
    history = st.session_state.setdefault('planning_conversation', [])
    for turn in history:
        with st.chat_message(turn['role']): st.write(turn['content'])
    entry = st.chat_input('Describe the networking application or answer the latest question')
    if entry:
        if history and history[-1]['role'] == 'user':
            history[-1]['content'] += '\n' + entry
        else:
            history.append({'role':'user','content':entry})
        if on_kaggle:
            try:
                if 'planning_backend' not in st.session_state:
                    st.session_state['planning_backend'] = KaggleAdapterBackend(ROOT/'models')
                reply = SMSACPipeline(st.session_state['planning_backend']).clarify(history)
                history.append({'role':'assistant','content':reply})
            except (InferenceError, ValueError) as error:
                st.error(str(error))
        else:
            st.info('Your notes are saved in this session. Live model replies require the trained adapters running on Kaggle.')
        with st.chat_message('user'): st.write(entry)
        if history[-1]['role'] == 'assistant':
            with st.chat_message('assistant'): st.write(history[-1]['content'])
    if history:
        st.download_button('Export discussion for Kaggle',json.dumps(history,ensure_ascii=False,indent=2),
                           file_name='conversation.json',mime='application/json')
        provisional = st.checkbox('Create a provisional draft with unresolved questions')
        if st.button('Create planning Word document',disabled=not on_kaggle):
            try:
                if 'planning_backend' not in st.session_state:
                    st.session_state['planning_backend'] = KaggleAdapterBackend(ROOT/'models')
                result = SMSACPipeline(st.session_state['planning_backend'],output_dir=ROOT/'outputs/planning').design(
                    history[0]['content'], conversation=history, provisional=provisional)
                st.session_state['planning_docx'] = result.docx_path.read_bytes()
            except (InferenceError, ValueError) as error: st.error(str(error))
        if 'planning_docx' in st.session_state:
            st.download_button('Download planning Word document',st.session_state['planning_docx'],file_name='planning.docx')
            if st.button('Recommend lifecycle from this document',disabled=not ON_KAGGLE):
                try:
                    result=trained_pipeline().recommend(st.session_state['planning_docx'],filename='planning.docx')
                    st.session_state['lifecycle_docx']=result.docx_path.read_bytes()
                    st.session_state['lifecycle_markdown']=result.markdown
                except (InferenceError,ValueError,DocumentError) as error: st.error(str(error))
            if 'lifecycle_docx' in st.session_state:
                st.download_button('Download explained lifecycle recommendation',st.session_state['lifecycle_docx'],file_name='lifecycle.docx')
                st.markdown(st.session_state['lifecycle_markdown'])
with demo:
    st.subheader('Campus observability demonstration')
    st.write(DEMO_REQUEST)
    if st.button('Generate demonstration reports',type='primary'):
        with st.spinner('Creating Word documents on CPU…'):
            with tempfile.TemporaryDirectory() as folder:
                results=SMSACPipeline(DemoBackend(),output_dir=folder).run_demo()
                st.session_state['demo_results']=[{'task':r.task,'markdown':r.markdown,'docx':r.docx_path.read_bytes()} for r in results]
    for result in st.session_state.get('demo_results',[]):
        st.subheader('Software design' if result['task']=='design' else 'Lifecycle recommendation')
        st.download_button('Download Word report',result['docx'],file_name=result['task']+'_DEMO.docx',mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',key=result['task'])
        with st.expander('Read the report'):
            st.markdown(result['markdown'])
with document:
    st.write('Inspect the text that the second model will receive. Upload a .docx file with decision evidence in body paragraphs and tables.')
    uploaded=st.file_uploader('Software design Word document',type=['docx'])
    if uploaded:
        try:
            result=read_design_docx(uploaded.getvalue(),filename=uploaded.name)
            for warning in result.warnings:st.warning(warning)
            st.text_area('Extracted text in document order',result.text,height=420)
            st.caption(f'{result.block_count} blocks · {len(result.text):,} characters. Actual model token limits are checked in Kaggle.')
            if st.button('Analyze uploaded document',disabled=not ON_KAGGLE):
                try:
                    recommendation=trained_pipeline().recommend(uploaded.getvalue(),filename=uploaded.name)
                    st.markdown(recommendation.markdown)
                    st.download_button('Download document analysis',recommendation.docx_path.read_bytes(),file_name='sdlc_analysis.docx')
                except (InferenceError,ValueError) as error: st.error(str(error))
        except DocumentError as error:st.error(str(error))
with training:
    st.write('The fresh LLM 2 training notebook uses the detailed SDLC dataset in Kaggle. A separate Kaggle chat notebook accepts the first model’s Word report, discusses decision factors, and exports a detailed SDLC Word document. No model weights are loaded on this laptop.')
    llm2_notebook=ROOT/'kaggle/train_llm2_detailed.ipynb'
    if llm2_notebook.exists():st.download_button('Download fresh LLM 2 training notebook',llm2_notebook.read_bytes(),file_name=llm2_notebook.name)
    llm2_chat=ROOT/'kaggle/llm2_chat_app.ipynb'
    if llm2_chat.exists():st.download_button('Download LLM 2 chat and Word notebook',llm2_chat.read_bytes(),file_name=llm2_chat.name)
    notebook=ROOT/'kaggle/train_dual_models.ipynb'
    if notebook.exists():st.download_button('Download training notebook',notebook.read_bytes(),file_name=notebook.name)
    inference_notebook=ROOT/'kaggle/use_dual_models.ipynb'
    if inference_notebook.exists():st.download_button('Download interactive discussion notebook',inference_notebook.read_bytes(),file_name=inference_notebook.name)
    manifest=ROOT/'data/dual/dataset_manifest.json'
    if manifest.exists():
        import json
        info=json.loads(manifest.read_text(encoding='utf-8'))
        st.metric('Weak-supervision training and evaluation rows',f"{info['total_rows']:,}")
        st.caption('Template-generated data requires expert review; row count is not evidence of model quality.')
    st.page_link('https://www.kaggle.com/code',label='Open Kaggle')

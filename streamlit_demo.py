import os

import numpy as np
import streamlit as st
from transformers import BertTokenizer, BertForQuestionAnswering

from answer import Answer
from download_st_helper import download_button
from eval_utils import MODELS, models_names_to_models_path, ALEPHBERT_PARASHOOT, MBERT_PARASHOOT, \
    LEGAL_QA_EXAMPLES_PATH, predict, save_results_to_docx, get_list_of_epochs_of_model, read_file_to_pars
from model_ckpt import ModelCkpt
from utils import load_jsonl_data, get_data_points_from_paragraphs, small_get_data_points_from_paragraphs

DEFAULT_TH = 35
TEXT = 'Text'
DOCUMENT = 'Document'


def create_hebrew_style():
    st.markdown("""
<style>
input {
  unicode-bidi:bidi-override;
  direction: RTL;
}
</style>
    """, unsafe_allow_html=True)


def create_titles():
    st.title("Hebrew question answering demo")
    st.markdown("## How to use the App?")
    st.markdown("This very simple. Fill the ``question`` and ``context`` text inputs and click"
                " the button ``run``.")


def error_style(error_msg):
    st.markdown(f"<h1 style='text-align: center; color: red;font-size: 22px;'>{error_msg}</h1>", unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_model_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    return model, tokenizer


def predict_on_one_segment(question, context, model_ckpt, top_k, progress=True):
    if progress:
        model_load_state = st.text("Loading model...")
    model, tokenizer = load_model_tokenizer(model_ckpt.path)
    if progress:
        model_load_state.text("Loading model...done!")
        tokenizer_load_state = st.text("Applying tokenizer...")
    encoding = tokenizer.encode_plus(question, context)
    if progress:
        tokenizer_load_state.text("Tokenization ... done!")
        model_apply_state = st.text("Predicting ...")
    answers = predict(model, tokenizer, encoding, top_k)
    if progress:
        model_apply_state.empty()
    return answers


def predict_on_many_segments(question, contexts, model_ckpt, top_k, th):
    answers_after_th = {key: [] for key in range(top_k)}

    th_to_prob = th / 100
    for i, context in enumerate(contexts):
        model_apply_state = st.text(f"Predicting answer on segment {i}/{len(contexts)}")
        answers = predict_on_one_segment(question, context, model_ckpt, top_k, progress=False)
        model_apply_state.empty()
        for j in range(len(answers)):
            if answers[j].answer_start_confidence > th_to_prob and answers[j].answer_end_confidence > th_to_prob:
                if j in answers_after_th:
                    answers_after_th[j].append(answers[j])
    return answers_after_th


def run_models(models_ckpts, question, context, top_k, context_input_type, th=0) -> np.array(Answer):
    examples_bottoms = st.columns(len(models_ckpts))
    models_answers = []
    for i, model_ckpt in enumerate(models_ckpts):
        with examples_bottoms[i]:
            if context_input_type == TEXT:
                answers = predict_on_one_segment(question, context, model_ckpt, top_k)
            else:
                answers = predict_on_many_segments(question, context, model_ckpt, top_k, th)

            models_answers.append(answers)
    return models_answers


def save_to_docx(question, answers, confidences, model_name):
    output_path = save_results_to_docx(question, answers, confidences, model_name)

    with open(output_path, 'rb') as f:
        s = f.read()

    download_title_for_displaying = download_button(s, output_path, f'Click here to download {model_name}.docx')
    st.markdown(download_title_for_displaying, unsafe_allow_html=True)


def display_significant_answer(answer, display_confidence=False):
    prefix, real_answer, suffix = answer.beginning_of_context_until_answer, answer.answer_itself, answer.end_of_context_from_the_answer
    st.markdown(f"<p style='text-align: input {{unicode-bidi:bidi-override; direction: RTL;}}"
                f" direction: RTL; color: grey; '>{prefix} <span style=font-weight:bold;>{real_answer} </span>{suffix}</p>",
                unsafe_allow_html=True)
    if display_confidence:
        st.markdown(f"<h6 style='text-align: center; color:red;'>sum of confidence: {'%.2f' % (answer.answer_start_confidence +  answer.answer_end_confidence)}</h6>",
        unsafe_allow_html=True)



def display_answer(answer, model_ckpt, display_no_find_answer=True):
    st.markdown(f"**_model {model_ckpt.name} - {model_ckpt.epoch}_**")
    st.markdown(
        f"<h6 style='text-align: center; color:red;'>confidence ({'%.2f' % answer.answer_start_confidence}, {'%.2f' % answer.answer_end_confidence})</h6>",
        unsafe_allow_html=True)
    if answer.significant_answer:
        display_significant_answer(answer)
    elif display_no_find_answer:
        st.markdown(f"**The model didn't predict an answer**")
    st.markdown('#')


def display_k_answers_on_file_continuously(question, model_ans, model_ckpt, answer_num):
    confs = []
    significant_answers = []
    st.markdown(f"**_model {model_ckpt.name} - {model_ckpt.epoch}_**")
    for answer in model_ans[answer_num]:
        if answer.significant_answer:
            display_significant_answer(answer, display_confidence=True)
            significant_answers.append(answer)
            confs.append(('%.2f' % answer.answer_start_confidence, '%.2f' % answer.answer_end_confidence))
    save_to_docx(question, significant_answers, confs, f"{model_ckpt.name}_{model_ckpt.epoch}")


def display_answers(question, models_answers, models_ckpts, top_k, context_input_type):
    for answer_k in range(top_k):
        st.markdown('#')
        st.markdown(f"<h6 style='text-align: center; color: blue;'>Answer number {answer_k + 1}:</h6>",
                    unsafe_allow_html=True)
        examples_bottoms = st.columns(len(models_ckpts))
        for model_ind, model_ckpt in enumerate(models_ckpts):
            with examples_bottoms[model_ind]:
                if context_input_type == TEXT:
                    display_answer(models_answers[model_ind][answer_k], model_ckpt)
                else:
                    display_k_answers_on_file_continuously(question, models_answers[model_ind], model_ckpt, answer_k)


def run_models_and_display_answers(models_ckpts, question, context, top_k, th):
    context_input_type = TEXT if isinstance(context, str) else DOCUMENT
    models_answers = run_models(models_ckpts, question, context, top_k, context_input_type, th=th)
    display_answers(question, models_answers, models_ckpts, top_k, context_input_type)


def get_model_epochs(model_name):
    models_ckpts = []
    epochs_paths = get_list_of_epochs_of_model(model_name)
    chosen_epochs = st.multiselect("select epochs:",
                                   sorted(list(epochs_paths.keys()), key=lambda x: int(x.split("_")[1])))
    for chosen_epoch in chosen_epochs:
        model_ckpt = ModelCkpt(model_name, path=epochs_paths[chosen_epoch],
                               epoch=chosen_epoch)
        models_ckpts.append(model_ckpt)
    models_ckpts = sorted(models_ckpts, key=lambda x: int(x.epoch.split("_")[1]))
    return models_ckpts


def create_models_objects(models_names):
    models_ckpts = []
    for model_name in models_names:
        model_ckpt = ModelCkpt(model_name, path=models_names_to_models_path[model_name],
                               epoch=os.path.basename(os.path.normpath(model_name)))
        models_ckpts.append(model_ckpt)
    return models_ckpts


def create_sidebar():
    with st.sidebar:
        st.subheader("Configure the model")
        models_names = st.multiselect("Choose models:", MODELS)
        if len(models_names) == 1:
            models_ckpts = get_model_epochs(models_names[0])
        else:
            models_ckpts = create_models_objects(models_names)
        top_k = st.slider('Select the number of the top k answers', 1, 5, 5)
        st.write('top_k:', top_k)
    return models_ckpts, top_k


def display_examples_questions():
    with st.expander("examples"):
        data = load_jsonl_data(LEGAL_QA_EXAMPLES_PATH)
        examples_bottoms = st.columns(len(data))
        for i, bottom in enumerate(examples_bottoms):
            with bottom:
                if st.button(f'example {i + 1}'):
                    example = data[i]
                    st.session_state.context = example['context'].strip()
                    st.session_state.question = example['question']


def create_next_prev_button(paragraphs):
    prev_col, _, next_col = st.columns([1, 10, 1])
    with next_col:
        if st.button('Next'):
            st.session_state['par_num'] = (st.session_state['par_num'] + 1) % len(paragraphs)
    with prev_col:
        if st.button('Previous'):
            st.session_state['par_num'] = (st.session_state['par_num'] - 1) % len(paragraphs)
    context_input_text = paragraphs[st.session_state['par_num']]
    st.markdown(f"segment number {st.session_state['par_num']}/{len(paragraphs)}:")
    pars = context_input_text.splitlines()
    for par in pars:
        st.markdown(
            f"<p style='text-align: input {{unicode-bidi:bidi-override; direction: RTL;}} direction: RTL; color: grey; '>{par}</p>",
            unsafe_allow_html=True)
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    return context_input_text


def eval_on_file_or_pars(uploaded_file, tokenizer):
    origin_paragraphs = read_file_to_pars(uploaded_file)
    paragraphs = get_data_points_from_paragraphs(origin_paragraphs, tokenizer)
    method_button, th_button = st.columns(2)
    th = 0
    with method_button:
        context_input_option_name = st.radio("Would you like to eval all the file?",
                                             ('whole file', 'paragraphs'))
        if context_input_option_name == 'whole file':
            with th_button:
                th = st.slider('Select th', 0, 100, DEFAULT_TH)
    if context_input_option_name == 'whole file':
        context_input_text = paragraphs
    else:
        context_input_text = create_next_prev_button(paragraphs)

    return context_input_text, th


def add_context_data_type(context_content, tokenizer):
    context_input_option_name = st.radio("Would you like to find an answer to a question in a document or text?",
                                         ('Text', 'Document'))
    th = 0
    if context_input_option_name == 'Text':
        context_input_option_button = st.empty()
        context_input_text = context_input_option_button.text_area("enter your context:", context_content,
                                                                   max_chars=12000, height=300)
    else:
        context_input_text, context_input_option_button = None, None
        uploaded_file = st.file_uploader("Choose a *docx* file", type=['docx'])
        if uploaded_file:
            context_input_text, th = eval_on_file_or_pars(uploaded_file, tokenizer)
    return context_input_option_name, context_input_text, context_input_option_button, th


def clear_session_data():
    st.session_state.context = ""
    st.session_state.question = ""


def add_clear_button(context_content, question_content):
    if st.button('clear'):
        clear_session_data()
        question_content, context_content = update_contents()
    return context_content, question_content


def add_run_button(question, question_content, context_input_text, context_content, models_ckpts, top_k, th):
    if st.button('run'):
        if not question and not question_content:
            error_style("There is no question")
        elif not context_input_text and not context_content:
            error_style("There is no context input")
        run_models_and_display_answers(models_ckpts, question, context_input_text, top_k, th)


def update_contents():
    question_content = st.session_state.question
    context_content = st.session_state.context
    return question_content, context_content


def add_optional_questions():
    data = load_jsonl_data(LEGAL_QA_EXAMPLES_PATH)
    with st.expander("pre-defined questions"):
        for i in range(len(data)):
            example = data[i]
            if st.button(example['question']):
                st.session_state.question = example['question']


def build_dashboard():
    create_titles()
    models_ckpts, top_k = create_sidebar()
    st.write('enter your own text or use an example:')
    add_optional_questions()
    display_examples_questions()

    question = st.empty()
    question_content, context_content = update_contents()
    tokenizer = BertTokenizer.from_pretrained(models_names_to_models_path[ALEPHBERT_PARASHOOT])
    for model in models_ckpts:
        if MBERT_PARASHOOT in model.name:
            tokenizer = BertTokenizer.from_pretrained(models_names_to_models_path[MBERT_PARASHOOT])
            break

    context_input_option_name, context_input_text, context_input_option_button, th = add_context_data_type(
        context_content, tokenizer)
    context_content, question_content = add_clear_button(context_content, question_content)

    question = question.text_input("enter your question:", question_content, max_chars=2000)
    if context_input_option_name == 'Text':
        context_input_text = context_input_option_button.text_area("enter your context_input_text:", context_content,
                                                                   max_chars=2000,
                                                                   height=300)
    question_content, context_content = update_contents()
    add_run_button(question, question_content, context_input_text, context_content, models_ckpts, top_k, th)


def init_session_state():
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'context' not in st.session_state:
        st.session_state['context'] = ""
    if 'par_num' not in st.session_state:
        st.session_state['par_num'] = 0


def build_data_app():
    init_session_state()
    create_hebrew_style()
    build_dashboard()


if __name__ == '__main__':
    build_data_app()

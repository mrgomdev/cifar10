import glob

import torch

import plotly.express as px
import streamlit as st

from main import Cifar10Label

if __name__ == '__main__':
    confusion_matrix_paths = sorted(glob.glob('./confusion_matrix_*.pt'))

    if 'current_idx' not in st.session_state:
        st.session_state['current_idx'] = 0
    if st.button('Next'):
        st.session_state['current_idx'] = min((st.session_state['current_idx'] + 1), len(confusion_matrix_paths) - 1)
    if st.button('Previous'):
        st.session_state['current_idx'] = max((st.session_state['current_idx'] - 1), 0)
    confusion_matrix_path = confusion_matrix_paths[st.session_state['current_idx']]

    confusion_matrix = torch.load(confusion_matrix_path)
    label_names = [Cifar10Label(idx).name for idx in range(10)]
    figure = px.imshow(confusion_matrix, labels=dict(x="Predicted", y="True", color="Count"), x=label_names, y=label_names, zmax=300, zmin=100, height=1000)
    st.header(confusion_matrix_path)
    st.plotly_chart(figure)

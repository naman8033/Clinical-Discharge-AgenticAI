import os
import streamlit as st


def save_file(uploaded_file, folder):

    os.makedirs(folder, exist_ok=True)

    save_path = os.path.join(folder, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def document_upload_page():

    st.title("📄 Clinical Document Upload")

    st.write(
        "Upload all patient documents before starting the workflow."
    )

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    bills_folder = os.path.join(
        project_root,
        "Data",
        "incoming",
        "bills"
    )

    doctor_folder = os.path.join(
        project_root,
        "Data",
        "incoming",
        "doctor_reports"
    )

    lab_folder = os.path.join(
        project_root,
        "Data",
        "incoming",
        "lab_reports"
    )

    col1, col2 = st.columns(2)

    with col1:

        bill_file = st.file_uploader(
            "💰 Upload Bill",
            type=["pdf"],
            key="bill"
        )

        doctor_file = st.file_uploader(
            "👨‍⚕ Upload Doctor Report",
            type=["pdf"],
            key="doctor"
        )

    with col2:

        lab_file = st.file_uploader(
            "🧪 Upload Lab Report",
            type=["pdf"],
            key="lab"
        )

    st.divider()

    if st.button(
        "🚀 Continue",
        use_container_width=True
    ):

        if not bill_file:

            st.error("Please upload Bill.")
            st.stop()

        if not doctor_file:

            st.error("Please upload Doctor Report.")
            st.stop()

        if not lab_file:

            st.error("Please upload Lab Report.")
            st.stop()

        save_file(
            bill_file,
            bills_folder,
        )

        save_file(
            doctor_file,
            doctor_folder
        )

        save_file(
            lab_file,
            lab_folder
        )

        st.success(
            "All documents uploaded successfully."
        )

        st.balloons()

        st.info(
            "Ready to start LangGraph workflow."
        )

        # Later
        # graph.invoke(...)
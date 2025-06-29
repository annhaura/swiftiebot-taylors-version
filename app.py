import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# --- Konfigurasi Awal dan Input API Key ---
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Masukkan **Google API Key** Anda:", type="password")
if not api_key:
    st.warning("Mohon masukkan API Key Anda untuk melanjutkan.", icon="🔑")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Cache Sumber Daya (Data dan Vector DB) ---
@st.cache_resource
def load_data_and_init_db():
    csv_url = "https://raw.githubusercontent.com/annhaura/swiftiebot-taylors-version/main/taylor_swift_songs.csv"
    try:
        df = pd.read_csv(csv_url)
        if 'Lyrics' not in df.columns:
            st.error("Kolom 'Lyrics' tidak ditemukan dalam file CSV Anda. Pastikan nama kolomnya benar.")
            st.stop()
        df['Lyrics'] = df['Lyrics'].fillna("")
    except Exception as e:
        st.error(f"Gagal memuat data dari URL: {csv_url}. Error: {e}")
        st.stop()

    st.info("Membangun database vektor...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = []
    for _, row in df.iterrows():
        lyrics = row['Lyrics']
        content = (
            f"Judul Lagu: {row['Song Name']}. Album: {row['Album']}. Lirik: {lyrics}. "
            f"Fitur Audio: Danceability {row['Danceability']:.2f}, Energy {row['Energy']:.2f}, "
            f"Key {row['Key']}, Loudness {row['Loudness']:.2f}, Mode {row['Mode']}, "
            f"Speechiness {row['Speechiness']:.2f}, Acousticness {row['Acousticness']:.2f}, "
            f"Instrumentalness {row['Instrumentalness']:.2f}, Liveness {row['Liveness']:.2f}, "
            f"Valence {row['Valence']:.2f}, Tempo {row['Tempo']:.2f} BPM, Duration_ms {row['Duration_ms']}."
        )
        metadata = {
            "song_name": row['Song Name'],
            "album": row['Album'],
            "danceability": row['Danceability'],
            "energy": row['Energy'],
            "key": row['Key'],
            "loudness": row['Loudness'],
            "mode": row['Mode'],
            "speechiness": row['Speechiness'],
            "acousticness": row['Acousticness'],
            "instrumentalness": row['Instrumentalness'],
            "liveness": row['Liveness'],
            "valence": row['Valence'],
            "tempo": row['Tempo'],
            "duration_ms": row['Duration_ms'],
            "lyrics": lyrics
        }
        documents.append(Document(page_content=content, metadata=metadata))

    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    st.success("Database vektor FAISS berhasil dibuat!")
    return df, vectorstore, embeddings

df_songs, vectorstore_db, embeddings_model = load_data_and_init_db()
retriever = vectorstore_db.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

@tool
def find_songs_by_criteria(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "Maaf, saya tidak menemukan lagu yang relevan dengan kueri tersebut."
    results = []
    for doc in docs:
        song_name = doc.metadata.get('song_name', 'N/A')
        album = doc.metadata.get('album', 'N/A')
        results.append(f"• '{song_name}' dari album '{album}'")
    return "\n".join(results)

@tool
def get_song_details(song_name: str) -> str:
    song_data = df_songs[df_songs['Song Name'].str.contains(song_name, case=False, na=False)]
    if not song_data.empty:
        song = song_data.iloc[0]
        details = [
            f"**Judul:** {song['Song Name']}",
            f"**Album:** {song['Album']}",
            f"**Danceability:** {song['Danceability']:.2f}",
            f"**Energy:** {song['Energy']:.2f}",
            f"**Tempo:** {song['Tempo']:.2f} BPM",
            f"**Valence:** {song['Valence']:.2f}",
            f"\n**Lirik:**\n{song['Lyrics']}"
        ]
        return "\n".join(details)
    return f"Maaf, detail untuk lagu '{song_name}' tidak ditemukan."

@tool
def get_songs_by_album(album_name: str) -> str:
    album_data = df_songs[df_songs['Album'].str.contains(album_name, case=False, na=False)]
    if not album_data.empty:
        song_list = album_data['Song Name'].tolist()
        return "\n".join(song_list)
    return f"Maaf, album '{album_name}' tidak ditemukan."

@tool
def explain_song_meaning_or_context(song_name: str) -> str:
    song_data = df_songs[df_songs['Song Name'].str.contains(song_name, case=False, na=False)]
    if not song_data.empty:
        song = song_data.iloc[0]
        info_for_llm = (
            f"Judul: {song['Song Name']}\n"
            f"Album: {song['Album']}\n"
            f"Lirik: {song['Lyrics']}\n"
            f"Danceability: {song['Danceability']:.2f}\n"
            f"Energy: {song['Energy']:.2f}\n"
            f"Tempo: {song['Tempo']:.2f} BPM\n"
            f"Valence: {song['Valence']:.2f}"
        )
        prompt = f"""Berikan penjelasan makna dan konteks dari lagu Taylor Swift berikut berdasarkan data ini:\n{info_for_llm}\nJelaskan secara puitis dan emosional."""
        return llm.invoke(prompt).content
    return f"Maaf, lagu '{song_name}' tidak ditemukan."

@tool
def recommend_similar_songs(song_name: str) -> str:
    target_song_data = df_songs[df_songs['Song Name'].str.contains(song_name, case=False, na=False)]
    if target_song_data.empty:
        return f"Maaf, lagu '{song_name}' tidak ditemukan."
    target_song = target_song_data.iloc[0]
    features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    target_vec = target_song[features].values
    others = df_songs[df_songs['Song Name'].str.lower() != song_name.lower()]
    distances = [((target_vec - row[features].values) ** 2).sum()**0.5 for _, row in others.iterrows()]
    similar = sorted(zip(distances, others['Song Name'], others['Album']))[:3]
    return "\n".join([f"• '{name}' dari album '{album}'" for _, name, album in similar])

tools = [find_songs_by_criteria, get_song_details, get_songs_by_album, explain_song_meaning_or_context, recommend_similar_songs]

prompt = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah SwiftieBot. Ketika pengguna meminta informasi, segera jalankan tools yang relevan dan tampilkan hasilnya dalam gaya percakapan natural dan antusias. Jangan hanya menyatakan niat, langsung berikan hasil dari tool yang dipanggil. Jika pengguna bilang 'iya', 'yang tadi', 'lanjut', dsb., gunakan konteks percakapan sebelumnya."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.set_page_config(page_title="SwiftieBot 🎤", page_icon="💖", layout="centered")
st.markdown("""<h1 style="text-align: center; color: #d6336c;"> Your song finder... but make it <i>Taylor’s Version</i></h1><p style="text-align: center; font-size: 16px; color: #555;">I don’t know about you, but I’m feelin’... ready to help you explore the world of Taylor Swift! <br>From country curls to pop anthems to poetic heartbreaks — tanyakan apa saja 🎶✨</p><hr>""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Tanyakan sesuatu tentang lagu Taylor Swift..."):
    with st.chat_message("user"):
        st.markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("assistant"):
        with st.spinner("SwiftieBot sedang mencari jawabannya untukmu..."):
            try:
                response = agent_executor.invoke({
                    "input": prompt_input,
                    "chat_history": st.session_state.chat_history
                })
                assistant_response = response["output"]
                st.markdown(assistant_response)
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt_input),
                    AIMessage(content=assistant_response)
                ])
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                error_message = f"Maaf, terjadi kesalahan: `{e}`. Coba lagi nanti ya 💔"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

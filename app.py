import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS # Import FAISS dari langchain_community
from langchain.docstore.document import Document
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage # Import AIMessage, HumanMessage

# --- Konfigurasi Awal dan Input API Key ---

# Coba muat variabel lingkungan dari file .env (untuk pengembangan lokal)
load_dotenv()

# Ambil API Key dari Streamlit Secrets, atau minta input dari pengguna
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Masukkan **Google API Key** Anda:", type="password")

# Hentikan aplikasi jika API Key belum dimasukkan
if not api_key:
    st.warning("Mohon masukkan API Key Anda untuk melanjutkan.", icon="🔑")
    st.stop()

# Set API Key sebagai environment variable agar LangChain bisa menggunakannya
os.environ["GOOGLE_API_KEY"] = api_key

# --- Cache Sumber Daya (Data dan Vector DB) ---
@st.cache_resource
def load_data_and_init_db():
    """
    Memuat data lagu dari URL publik dan menginisialisasi database vektor FAISS.
    Database akan dibangun ulang setiap kali aplikasi dimulai/di-refresh.
    """
    # URL ini sudah disesuaikan dengan lokasi file Anda di GitHub
    csv_url = "https://raw.githubusercontent.com/annhaura/swiftiebot-taylors-version/main/taylor_swift_songs.csv" 

    try:
        df = pd.read_csv(csv_url)
        # Pastikan kolom 'Lyrics' ada dan tangani NaN jika ada
        if 'Lyrics' not in df.columns:
            st.error("Kolom 'Lyrics' tidak ditemukan dalam file CSV Anda. Pastikan nama kolomnya benar.")
            st.stop()
        df['Lyrics'] = df['Lyrics'].fillna("") # Ganti NaN dengan string kosong
    except Exception as e:
        st.error(f"Gagal memuat data dari URL: {csv_url}. Pastikan URL benar, bisa diakses publik, dan format CSV valid. Error: {e}")
        st.stop()
    
    st.info("Membangun database vektor (ini mungkin butuh waktu tergantung ukuran data)...")
    
    # Inisialisasi model embedding untuk mengubah teks menjadi vektor
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    documents = []
    for index, row in df.iterrows():
        lyrics = row['Lyrics']
        
        # Gabungkan semua informasi relevan, termasuk lirik, untuk konten dokumen
        content = (
            f"Judul Lagu: {row['Song Name']}. Album: {row['Album']}. Lirik: {lyrics}. "
            f"Fitur Audio: Danceability {row['Danceability']:.2f}, Energy {row['Energy']:.2f}, "
            f"Key {row['Key']}, Loudness {row['Loudness']:.2f}, Mode {row['Mode']}, "
            f"Speechiness {row['Speechiness']:.2f}, Acousticness {row['Acousticness']:.2f}, "
            f"Instrumentalness {row['Instrumentalness']:.2f}, Liveness {row['Liveness']:.2f}, "
            f"Valence {row['Valence']:.2f}, Tempo {row['Tempo']:.2f} BPM, Duration_ms {row['Duration_ms']}."
        )
        
        # Tambahkan semua metadata yang mungkin berguna
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
            "lyrics": lyrics # Simpan lirik penuh di metadata juga
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Buat FAISS database dari dokumen
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    st.success("Database vektor FAISS berhasil dibuat!")
    
    return df, vectorstore, embeddings

# Muat data dan inisialisasi database saat aplikasi dimulai
df_songs, vectorstore_db, embeddings_model = load_data_and_init_db()
retriever = vectorstore_db.as_retriever() # Digunakan oleh tools untuk pencarian

# --- Inisialisasi Model Bahasa Besar (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# --- Definisi Tools untuk Agent ---

@tool
def find_songs_by_criteria(query: str) -> str:
    """
    Mencari lagu Taylor Swift yang relevan berdasarkan kriteria luas seperti tema, mood, kata kunci lirik,
    atau karakteristik audio. Ini adalah implementasi RAG (Retrieval-Augmented Generation) utama.
    Ini akan mengembalikan daftar judul lagu dan albumnya.
    Gunakan tool ini ketika pengguna ingin menemukan lagu berdasarkan deskripsi umum.
    """
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "Maaf, saya tidak menemukan lagu yang relevan dengan kueri tersebut."
    
    results = []
    for doc in docs:
        song_name = doc.metadata.get('song_name', 'N/A')
        album = doc.metadata.get('album', 'N/A')
        results.append(f"• '{song_name}' dari album '{album}'")
    return "Saya menemukan beberapa lagu yang mungkin Anda suka:\n" + "\n".join(results)

@tool
def get_song_details(song_name: str) -> str:
    """
    Mendapatkan detail lengkap (lirik penuh, album, dan semua fitur audio seperti Danceability, Energy, Tempo, dll.)
    untuk lagu Taylor Swift tertentu.
    Gunakan tool ini ketika pengguna secara spesifik meminta detail tentang sebuah lagu.
    Input harus berupa nama lagu yang jelas dan akurat.
    """
    song_data = df_songs[df_songs['Song Name'].str.contains(song_name, case=False, na=False)]
    
    if not song_data.empty:
        song = song_data.iloc[0]
        details = []
        details.append(f"**Judul:** {song['Song Name']}")
        details.append(f"**Album:** {song['Album']}")
        details.append(f"**Danceability:** {song['Danceability']:.2f} (0.0-1.0)")
        details.append(f"**Energy:** {song['Energy']:.2f} (0.0-1.0)")
        details.append(f"**Key:** {song['Key']} (Nada Dasar)")
        details.append(f"**Loudness:** {song['Loudness']:.2f} dB")
        details.append(f"**Mode:** {'Major' if song['Mode'] == 1 else 'Minor'}")
        details.append(f"**Speechiness:** {song['Speechiness']:.2f} (0.0-1.0)")
        details.append(f"**Acousticness:** {song['Acousticness']:.2f} (0.0-1.0)")
        details.append(f"**Instrumentalness:** {song['Instrumentalness']:.2f} (0.0-1.0)")
        details.append(f"**Liveness:** {song['Liveness']:.2f} (0.0-1.0)")
        details.append(f"**Valence:** {song['Valence']:.2f} (0.0-1.0, seberapa positif/bahagia)")
        details.append(f"**Tempo:** {song['Tempo']:.2f} BPM")
        details.append(f"**Durasi:** {round(song['Duration_ms'] / 60000, 2)} menit")
        details.append(f"\n**Lirik:**\n{song['Lyrics']}") # Lirik penuh disertakan
        return "\n".join(details)
    return f"Maaf, detail untuk lagu '{song_name}' tidak ditemukan. Pastikan nama lagu sudah benar."

@tool
def get_songs_by_album(album_name: str) -> str:
    """
    Mendaftar semua lagu yang ada dalam album Taylor Swift tertentu.
    Gunakan tool ini ketika pengguna ingin melihat daftar lagu dari sebuah album.
    Input harus berupa nama album yang jelas dan akurat.
    """
    album_data = df_songs[df_songs['Album'].str.contains(album_name, case=False, na=False)]
    
    if not album_data.empty:
        song_list = album_data['Song Name'].tolist()
        return f"Lagu-lagu di album '{album_name}':\n• " + "\n• ".join(song_list)
    return f"Maaf, album '{album_name}' tidak ditemukan. Pastikan nama album sudah benar."

@tool
def explain_song_meaning_or_context(song_name: str) -> str:
    """
    Menjelaskan makna, tema mendalam, narasi, atau konteks di balik sebuah lagu Taylor Swift,
    sangat mengandalkan liriknya serta fitur audionya.
    Gunakan tool ini ketika pengguna meminta penjelasan atau interpretasi tentang sebuah lagu.
    Input harus berupa nama lagu yang jelas dan akurat.
    """
    song_data = df_songs[df_songs['Song Name'].str.contains(song_name, case=False, na=False)]
    
    if not song_data.empty:
        song = song_data.iloc[0]
        
        # Kumpulkan semua informasi relevan untuk diberikan ke LLM untuk penjelasan
        # Termasuk lirik penuh
        info_for_llm = (
            f"Judul: {song['Song Name']}\n"
            f"Album: {song['Album']}\n"
            f"Lirik: {song['Lyrics']}\n" # Lirik penuh
            f"Danceability: {song['Danceability']:.2f}\n"
            f"Energy: {song['Energy']:.2f}\n"
            f"Tempo: {song['Tempo']:.2f} BPM\n"
            f"Valence: {song['Valence']:.2f} (seberapa positif/bahagia suasana lagu)"
        )

        # Prompt LLM untuk menjelaskan berdasarkan lirik dan fitur audio
        explanation_prompt = f"""Berdasarkan informasi berikut tentang lagu Taylor Swift '{song['Song Name']}':

{info_for_llm}

Tolong jelaskan makna, tema utama, narasi, atau konteks di balik lagu ini. Bagaimana lirik dan fitur audionya berkontribusi pada pesan atau suasana lagu? Jelaskan dengan gaya bahasa Taylor Swift yang puitis, penuh perasaan, dan mendalam. Jika lagu tersebut memiliki cerita yang kuat, ceritakan secara singkat."""
        
        explanation = llm.invoke(explanation_prompt).content
        return f"Tentu, mari kita selami makna lagu '{song['Song Name']}':\n\n{explanation}"
    
    return f"Maaf, lagu '{song_name}' tidak ditemukan. Tidak dapat menjelaskan maknanya."

@tool
def recommend_similar_songs(song_name: str) -> str:
    """
    Merekomendasikan lagu Taylor Swift lain yang serupa berdasarkan fitur audio (Danceability, Energy, Tempo, dll.).
    Gunakan tool ini ketika pengguna meminta rekomendasi lagu yang mirip.
    Input harus berupa nama lagu yang jelas dan akurat.
    """
    target_song_data = df_songs[df_songs['Song Name'].str.contains(song_name, case=False, na=False)]

    if target_song_data.empty:
        return f"Maaf, lagu '{song_name}' tidak ditemukan. Tidak dapat memberikan rekomendasi."

    target_song = target_song_data.iloc[0]
    # Sertakan semua fitur numerik yang relevan untuk perbandingan kemiripan
    numerical_features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 
                          'Speechiness', 'Acousticness', 'Instrumentalness', 
                          'Liveness', 'Valence', 'Tempo']
    target_features = target_song[numerical_features].values

    df_other_songs = df_songs[df_songs['Song Name'].str.lower() != song_name.lower()]
    
    if df_other_songs.empty:
        return "Tidak ada lagu lain untuk dibandingkan."

    distances = []
    for idx, row in df_other_songs.iterrows():
        other_features = row[numerical_features].values
        distance = ((target_features - other_features)**2).sum()**0.5 # Euclidean distance
        distances.append((distance, row['Song Name'], row['Album']))
    
    distances.sort(key=lambda x: x[0])

    top_recommendations = distances[:3]

    if not top_recommendations:
        return f"Tidak dapat menemukan rekomendasi lagu yang mirip dengan '{song_name}'."
    
    rec_list = []
    for dist, rec_song_name, rec_album in top_recommendations:
        rec_list.append(f"• '{rec_song_name}' dari album '{rec_album}'")

    return "Lagu-lagu yang mirip:\n" + "\n".join(rec_list)

# Daftar semua tools yang akan digunakan oleh agent
tools = [find_songs_by_criteria, get_song_details, get_songs_by_album, explain_song_meaning_or_context, recommend_similar_songs]

# --- Prompt untuk Agent ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah SwiftieBot, chatbot pencari lagu Taylor Swift yang berpengetahuan luas, ramah, dan antusias.
    Tujuanmu adalah membantumu menemukan dan mempelajari tentang lagu-lagu Taylor Swift dengan semangat seorang Swiftie sejati.
    Kamu memiliki akses ke data lengkap lagu Taylor Swift, termasuk lirik dan fitur audio.
    Kamu dapat:
    1. Mencari lagu berdasarkan tema, mood, kata kunci lirik, atau karakteristik audio.
    2. Memberikan detail lengkap (lirik penuh, album, dan semua fitur audio) dari lagu tertentu.
    3. Mendaftar semua lagu dari album tertentu.
    4. Menjelaskan makna, tema mendalam, narasi, atau konteks di balik sebuah lagu berdasarkan lirik dan fitur audionya.
    5. Merekomendasikan lagu lain yang mirip berdasarkan fitur audio dari sebuah lagu.

    Selalu berusaha memberikan jawaban yang informatif, akurat, dan membantu.
    Jika tidak yakin, tanyakan informasi lebih lanjut atau berikan opsi lain.
    **SANGAT PENTING: Ketika kamu menggunakan salah satu tool, jangan pernah mencetak atau menampilkan nama tool atau pemanggilan fungsi Python secara langsung di outputmu. Ambil hasil (output) dari tool yang sudah dieksekusi, lalu rumuskan hasil tersebut ke dalam kalimat yang koheren, natural, ramah pengguna, dan penuh semangat Swiftie. Integrasikan informasi tersebut secara mulus ke dalam responsmu seolah-olah kamu mengetahuinya secara langsung.**
    **Perhatikan riwayat percakapan sebelumnya untuk memahami konteks kueri pengguna, terutama jika kueri singkat dan merujuk pada topik yang baru saja dibahas. Gunakan informasi dari riwayat untuk melengkapi kueri singkat menjadi kueri lengkap untuk tool jika memungkinkan.**
    Berinteraksilah dengan ramah dan penuh semangat layaknya seorang penggemar sejati Taylor Swift!"""), # <--- Perhatikan bagian yang di-bold dan kapital
    MessagesPlaceholder(variable_name="chat_history"), # <--- Menambahkan placeholder untuk riwayat obrolan
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad") # <--- Menggunakan agent_scratchpad
])

# Buat agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Buat AgentExecutor (ini yang akan menjalankan agent dan tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Antarmuka Pengguna Streamlit ---
st.set_page_config(page_title="SwiftieBot", page_icon="💖")
st.title("🎤 Your song finder... but make it Taylor's Version")
st.caption(
    "I don’t know about you, but I’m feelin’... ready to help you explore the world of Taylor Swift! 💫\n"
    "From country curls to pop anthems to poetic heartbreaks — tanyakan apa saja 🎤✨"
)

# Inisialisasi riwayat obrolan di session state Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
# Inisialisasi riwayat obrolan untuk LangChain Agent
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Tampilkan pesan obrolan dari riwayat setiap kali aplikasi di-rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Reaksi terhadap input pengguna
if prompt_input := st.chat_input("Tanyakan sesuatu tentang lagu Taylor Swift..."):
    # Tampilkan pesan pengguna di kontainer obrolan
    with st.chat_message("user"):
        st.markdown(prompt_input)
    
    # Tambahkan pesan pengguna ke riwayat obrolan Streamlit untuk ditampilkan
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    with st.chat_message("assistant"):
        with st.spinner("SwiftieBot sedang mencari jawabannya untukmu..."):
            try:
                # Panggil agent executor, berikan input dan chat_history
                response = agent_executor.invoke({
                    "input": prompt_input,
                    "chat_history": st.session_state.chat_history # Meneruskan riwayat obrolan
                })
                assistant_response = response["output"]
                st.markdown(assistant_response)
                
                # Tambahkan pesan pengguna dan respons asisten ke riwayat obrolan LangChain (untuk konteks)
                st.session_state.chat_history.append(HumanMessage(content=prompt_input))
                st.session_state.chat_history.append(AIMessage(content=assistant_response))
                
                # Tambahkan respons asisten ke riwayat obrolan Streamlit untuk ditampilkan
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                # Tangani error jika terjadi
                error_message = f"Maaf, sepertinya ada kesalahan saat mencari. Terjadi masalah: {e}. Bisakah Anda coba lagi dengan pertanyaan yang berbeda?"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

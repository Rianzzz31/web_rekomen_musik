import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.markdown("""
<style>
/* Background utama */
.stApp {
    background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #6C63FF;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Tombol */
.stButton > button {
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.4em 1em;
}
.stButton > button:hover {
    background-color: #554fd8;
}

/* Card lagu */
div[data-testid="stVerticalBlock"] > div:has(h3) {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# CONFIG
# ===========================
st.set_page_config("Hybrid Music Recommender", layout="wide", page_icon="🎧")
DATA_PATH = "spotify_data_clean.csv"
USER_PATH = "users.csv"
PREF_PATH = "user_preferences.csv"
PLAYED_PATH = "user_played.csv"
RATING_PATH = "user_ratings.csv"

# ===========================
# LOAD MUSIC (ANTI DUPLIKAT)
# ===========================
@st.cache_data
def load_music():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset="track_name").reset_index(drop=True)

    FEATURES = [
        "danceability","energy","valence",
        "tempo","acousticness","liveness","speechiness"
    ]
    FEATURES = [f for f in FEATURES if f in df.columns]

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    df[FEATURES] = MinMaxScaler().fit_transform(df[FEATURES])
    return df, FEATURES

df_music, FEATURES = load_music()

# ===========================
# FILE INIT
# ===========================
for path, cols in [
    (USER_PATH, ["username","password"]),
    (PREF_PATH, ["username","track_name","timestamp"]),
    (PLAYED_PATH, ["username","track_name","timestamp"]),
    (RATING_PATH, ["username","track_name","rating","timestamp"])
]:
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path,index=False)

# ===========================
# SESSION
# ===========================
if "page" not in st.session_state: st.session_state.page="login"
if "user" not in st.session_state: st.session_state.user=None
if "home_recs" not in st.session_state: st.session_state.home_recs=None
if "need_update_home" not in st.session_state:
    st.session_state.need_update_home = False

# ===========================
# DATA UTILS
# ===========================
def save_fav(u,t):
    df=pd.read_csv(PREF_PATH)
    df.loc[len(df)] = [u,t,pd.Timestamp.now()]
    df.to_csv(PREF_PATH,index=False)

def save_played(u,t):
    df=pd.read_csv(PLAYED_PATH)
    df.loc[len(df)] = [u,t,pd.Timestamp.now()]
    df.to_csv(PLAYED_PATH,index=False)

def save_rating(u,t,r):
    df=pd.read_csv(RATING_PATH)
    df.loc[len(df)] = [u,t,r,pd.Timestamp.now()]
    df.to_csv(RATING_PATH,index=False)

def get_user_fav(u):
    return pd.read_csv(PREF_PATH).query("username==@u")

# ===========================
# COLLABORATIVE FILTERING
# ===========================
def user_item():
    df = pd.read_csv(RATING_PATH)
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(
        index="username",
        columns="track_name",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )

def cf_score(u):
    m = user_item()
    if m.empty or u not in m.index or m.shape[0] < 2:
        return pd.Series(dtype=float)
    sim = cosine_similarity(m)
    sim = pd.DataFrame(sim,index=m.index,columns=m.index)
    return m.T.dot(sim[u])

# ===========================
# HYBRID RECOMMENDER (0.5 + 0.5)
# ===========================
def hybrid_recommend(u,n=10):
    fav = get_user_fav(u)
    if fav.empty:
        out = df_music.sample(n)
        out["hybrid_score"] = 0
        return out

    fav = fav.merge(df_music,on="track_name",how="left")
    user_vec = fav[FEATURES].mean().values.reshape(1,-1)

    cbf = cosine_similarity(user_vec,df_music[FEATURES])[0]
    cbf = (cbf-cbf.min())/(cbf.max()-cbf.min()+1e-9)

    df = df_music.copy()
    df["cbf_score"] = cbf

    cf = cf_score(u).reset_index()
    cf.columns=["track_name","cf_score"]
    df = df.merge(cf,on="track_name",how="left").fillna(0)

    if df.cf_score.max()>0:
        df.cf_score /= df.cf_score.max()

    df["hybrid_score"] = 0.5*df.cbf_score + 0.5*df.cf_score

    return (
        df[~df.track_name.isin(fav.track_name)]
        .sort_values("hybrid_score",ascending=False)
        .head(n)
    )

# ===========================
# LOGIN & REGISTER
# ===========================
def login_page():
    st.title("🔐 Login")
    u=st.text_input("Username")
    p=st.text_input("Password",type="password")

    if st.button("Login"):
        df=pd.read_csv(USER_PATH)
        if u in df.username.values and df.loc[df.username==u,"password"].values[0]==p:
            st.session_state.user=u
            st.session_state.page="initial" if get_user_fav(u).empty else "home"
            st.rerun()
        else:
            st.error("Login gagal")

    if st.button("Daftar"):
        st.session_state.page="register"
        st.rerun()

def register_page():
    st.title("📝 Register")
    u=st.text_input("Username")
    p=st.text_input("Password",type="password")
    if st.button("Buat Akun"):
        df=pd.read_csv(USER_PATH)
        if u not in df.username.values:
            df.loc[len(df)] = [u,p]
            df.to_csv(USER_PATH,index=False)
            st.success("Akun dibuat")
            st.session_state.page="login"
            st.rerun()

# ===========================
# INITIAL
# ===========================
def initial_page():
    st.title("✨ Pilih Lagu Favorit Awal")
    tracks=df_music.sample(20,random_state=42).track_name.tolist()
    with st.form("init"):
        sel=st.multiselect("Pilih minimal 3 lagu",tracks)
        ok=st.form_submit_button("Simpan")
    if ok and len(sel)>=3:
        for t in sel: save_fav(st.session_state.user,t)
        st.session_state.page="home"
        st.rerun()

# ===========================
# HOME (TIDAK AUTO REFRESH)
# ===========================
def home_page():
    st.title("🎧 Rekomendasi Hybrid")

    if st.session_state.home_recs is None:
        st.session_state.home_recs = hybrid_recommend(st.session_state.user,10)

    for i,r in st.session_state.home_recs.iterrows():
        st.markdown(f"### 🎵 {r.track_name}")
        rating=st.slider("Rating",1,5,3,key=f"hr{i}")

        c1,c2,c3=st.columns(3)
        with c1:
            if st.button("▶ Putar",key=f"hp{i}"):
                save_played(st.session_state.user,r.track_name)
        with c2:
            if st.button("⭐ Simpan Rating",key=f"hr2{i}"):
                save_rating(st.session_state.user,r.track_name,rating)
        with c3:
            if st.button("➕ Simpan Lagu",key=f"hs{i}"):
                save_fav(st.session_state.user,r.track_name)

        st.divider()

# ===========================
# SEARCH (MEMICU UPDATE HOME)
# ===========================
def search_page():
    st.title("🔍 Cari Lagu")
    q=st.text_input("Cari Lagu")
    if not q: return

    res=df_music[df_music.track_name.str.contains(q,case=False)]
    for i,r in res.iterrows():
        st.markdown(f"### 🎵 {r.track_name}")
        rating=st.slider("Rating",1,5,3,key=f"sr{i}")

        c1,c2,c3=st.columns(3)
        with c1:
            if st.button("▶ Putar",key=f"sp{i}"):
                save_played(st.session_state.user,r.track_name)
                st.session_state.need_update_home = True
        with c2:
            if st.button("⭐ Simpan Rating",key=f"sr2{i}"):
                save_rating(st.session_state.user,r.track_name,rating)
                st.session_state.need_update_home = True
        with c3:
            if st.button("➕ Simpan Lagu",key=f"ss{i}"):
                save_fav(st.session_state.user,r.track_name)
                st.session_state.need_update_home = True

        st.divider()

# ===========================
# FAVORITE
# ===========================
def favorite_page():
    st.title("💾 Lagu Disimpan")
    fav=get_user_fav(st.session_state.user)
    for i,r in fav.iterrows():
        if st.button(f"▶ {r.track_name}",key=f"fp{i}"):
            save_played(st.session_state.user,r.track_name)

# ===========================
# EVALUATION
# ===========================
def eval_page():
    st.title("📊 Evaluasi Sistem")

    ratings = pd.read_csv(RATING_PATH)
    rows = []
    rmses = []

    for u in ratings.username.unique():
        ur = ratings[ratings.username == u]
        if len(ur) < 3:
            continue

        rec = hybrid_recommend(u, 10)
        hit = len(set(ur.track_name) & set(rec.track_name))

        rows.append({
            "User": u,
            "Precision@10": hit / 10,
            "Recall@10": hit / len(ur)
        })

        m = ur.merge(rec[["track_name", "hybrid_score"]], on="track_name")
        if not m.empty:
            pred = m.hybrid_score * 4 + 1
            rmses.append(np.sqrt(((pred - m.rating) ** 2).mean()))

    if not rows:
        st.info("Belum cukup data rating")
        return

    # =========================
    # TABEL EVALUASI
    # =========================
    df_eval = pd.DataFrame(rows)
    st.dataframe(df_eval, use_container_width=True)

    # =========================
    # GRAFIK PRECISION & RECALL
    # =========================
    fig, ax = plt.subplots()
    ax.plot(df_eval.User, df_eval["Precision@10"], marker="o", label="Precision")
    ax.plot(df_eval.User, df_eval["Recall@10"], marker="o", label="Recall")
    ax.legend()
    st.pyplot(fig)

    # =========================
    # RMSE (ANGKA + KATEGORI)
    # =========================
    if rmses:
        rmse_mean = round(np.mean(rmses), 2)

        if rmse_mean < 1:
            label = "Akurat"
            box = st.success
        elif rmse_mean < 2:
            label = "Cukup Akurat"
            box = st.warning
        else:
            label = "Kurang Akurat"
            box = st.error

        st.metric("RMSE (Root Mean Squared Error)", rmse_mean)
        box(f"Kategori Akurasi Sistem: **{label}**")

# ===========================
# SIDEBAR (TIDAK DIUBAH)
# ===========================
if st.session_state.user:
    st.sidebar.success(st.session_state.user)
    if st.sidebar.button("🏠 Home"):
        if st.session_state.need_update_home:
            st.session_state.home_recs = None
            st.session_state.need_update_home = False
        st.session_state.page="home"

    if st.sidebar.button("🔍 Search"): st.session_state.page="search"
    if st.sidebar.button("💾 Favorit"): st.session_state.page="favorite"
    if st.sidebar.button("📊 Evaluasi"): st.session_state.page="eval"

    st.sidebar.divider()
    st.sidebar.markdown("### ▶ Lagu Disimpan")
    fav=get_user_fav(st.session_state.user)
    for i,r in fav.iterrows():
        if st.sidebar.button(r.track_name,key=f"sb{i}"):
            save_played(st.session_state.user,r.track_name)

    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.session_state.page="login"
        st.rerun()

# ===========================
# ROUTER
# ===========================
if st.session_state.page=="login": login_page()
elif st.session_state.page=="register": register_page()
elif st.session_state.page=="initial": initial_page()
elif st.session_state.page=="home": home_page()
elif st.session_state.page=="search": search_page()
elif st.session_state.page=="favorite": favorite_page()
elif st.session_state.page=="eval": eval_page()

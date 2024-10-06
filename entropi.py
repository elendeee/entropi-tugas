import streamlit as st
import matplotlib.pyplot as plt
from math import log2

def cross_entropy(p, q):
    return -sum([p[i]*log2(q[i]) for i in range(len(p))])

p = [0.35, 0.45, 0.20]  
q = [0.30, 0.50, 0.20]  

ce_pq = cross_entropy(p, q)

st.title("Tugas Data Science")
st.write("Nama: Elen Debina")
st.write("NPM: 227006043")
st.write("Kelas: A")

st.subheader("Prediksi Tingkat Pendidikan Berdasarkan Data Ekonomi")
st.write(f"Entropi Silang (H(P, Q)): {ce_pq:.3f} bits")

st.subheader("Perbandingan Distribusi Pendidikan")
categories = ['Pendidikan Dasar', 'Pendidikan Menengah', 'Pendidikan Tinggi']

fig, ax = plt.subplots()
ax.bar(categories, p, width=0.4, label='Data Sebenarnya', align='center', alpha=0.7)
ax.bar(categories, q, width=0.4, label='Prediksi Model', align='edge', alpha=0.7)
ax.set_ylabel('Probabilitas')
ax.set_title('Perbandingan Distribusi Pendidikan')
ax.legend()

st.pyplot(fig)

# Penjelasan
st.subheader("Penjelasan:")
st.write("""
Entropi silang H(P, Q) menunjukkan perbedaan antara distribusi prediksi model
dan data sebenarnya. Semakin besar nilai entropi silang, semakin besar kesalahan
model dalam memprediksi data sebenarnya. Dalam kasus ini, distribusi prediksi
model cukup mendekati distribusi sebenarnya, namun ada sedikit perbedaan di tingkat
pendidikan dasar dan menengah.
""")

// firebase.js
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.11.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.11.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.11.0/firebase-firestore.js";

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyC9PeGYw_ka3l207qm3fCoWGGoJqAk4RNQ",
    authDomain: "smartclassroomplatform.firebaseapp.com",
    projectId: "smartclassroomplatform",
    storageBucket: "smartclassroomplatform.firebasestorage.app",
    messagingSenderId: "1069513538420",
    appId: "1:1069513538420:web:b2bbffe1d07f3aadac0a21",
    measurementId: "G-03QQ8CHPGL"
  };

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);

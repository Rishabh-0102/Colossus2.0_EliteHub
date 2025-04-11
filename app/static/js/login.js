// login.js
import { auth, db } from "./firebase.js";
import {
  signInWithEmailAndPassword,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/10.11.0/firebase-auth.js";
import {
  doc,
  getDoc
} from "https://www.gstatic.com/firebasejs/10.11.0/firebase-firestore.js";

const form = document.getElementById("login-form");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    const uid = userCredential.user.uid;

    const userDoc = await getDoc(doc(db, "users", uid));
    if (!userDoc.exists()) {
      alert("Unauthorized user or missing user role data.");
      return;
    }

    const userData = userDoc.data();
    localStorage.setItem("userName", userData.name || "User");
    localStorage.setItem("userRole", userData.role);
    window.location.href = "dashboard";

  } catch (error) {
    alert(error.message);
    console.error("Firebase error:", error);
  }
});
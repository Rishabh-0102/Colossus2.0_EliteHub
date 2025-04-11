import { auth, db } from "./firebase.js";
import {
  onAuthStateChanged,
  signOut
} from "https://www.gstatic.com/firebasejs/10.11.0/firebase-auth.js";
import {
  doc, getDoc, collection, addDoc, getDocs, query, where, serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.11.0/firebase-firestore.js";

const welcomeText = document.getElementById("welcome-text");
const dataDisplay = document.getElementById("data-display");
const teacherOptions = document.getElementById("teacher-options");
const startAttendanceForm = document.getElementById("start-attendance-form");
const checkAnalysisSection = document.getElementById("check-analysis-section");
const analysisResult = document.getElementById("analysis-result");
const analysisDateInput = document.getElementById("analysis-date");

let currentUser = null;

// ---------------------- AUTH STATE ----------------------
onAuthStateChanged(auth, async (user) => {
  if (!user) {
    window.location.href = "/login";
    return;
  }

  currentUser = user;
  const uid = user.uid;
  const userDocRef = doc(db, "users", uid);
  const userDocSnap = await getDoc(userDocRef);

  if (!userDocSnap.exists()) {
    welcomeText.textContent = "User record not found.";
    return;
  }

  const userData = userDocSnap.data();
  const { name, role } = userData;

  welcomeText.textContent = `Welcome ${name}${role === "teacher" ? " Sir" : ""}`;

  if (role === "student") {
    const studentDataDoc = await getDoc(doc(db, "studentsData", uid));
    if (studentDataDoc.exists()) {
      const d = studentDataDoc.data();
      dataDisplay.innerHTML = `
        <p><strong>Your Data:</strong><br>
        Attendance: ${d.attendance ?? "Not recorded"}<br>
        Focus Score: ${d.focusScore ?? "Not recorded"}</p>`;
    } else {
      dataDisplay.innerHTML = `<p>No data found yet for your behavior analysis.</p>`;
    }
  }

  else if (role === "teacher") {
    teacherOptions.classList.remove("hidden");
  }

  else if (role === "organization" || role === "admin") {
    await showAdminDashboard();
  }
});

// ---------------------- ADMIN DASHBOARD ----------------------
async function showAdminDashboard() {
  dataDisplay.innerHTML = `<h3>All Teachers:</h3>`;
  const teacherQuery = query(collection(db, "users"), where("role", "==", "teacher"));
  const teacherSnapshot = await getDocs(teacherQuery);

  if (teacherSnapshot.empty) {
    dataDisplay.innerHTML += `<p>No teachers found.</p>`;
  } else {
    teacherSnapshot.forEach(docSnap => {
      const t = docSnap.data();
      dataDisplay.innerHTML += `<p><strong>${t.name} Sir</strong> (Teacher)</p>`;
    });
  }

  dataDisplay.innerHTML += `<h3>All Students:</h3>`;
  const studentSnapshot = await getDocs(collection(db, "studentsData"));

  if (studentSnapshot.empty) {
    dataDisplay.innerHTML += `<p>No students found.</p>`;
  } else {
    for (const docSnap of studentSnapshot.docs) {
      const s = docSnap.data();
      const studentId = docSnap.id;

      const studentUserDoc = await getDoc(doc(db, "users", studentId));
      const studentName = studentUserDoc.exists() ? studentUserDoc.data().name : "Unnamed";

      dataDisplay.innerHTML += `
        <p><strong>${studentName}</strong>: 
        Attendance: ${s.attendance || "N/A"}, 
        Focus Score: ${s.focusScore || "N/A"}</p>`;
    }
  }
}

// --------------------- SHOW / HIDE SECTIONS ---------------------
window.showStartAttendance = () => {
  startAttendanceForm.classList.remove("hidden");
  checkAnalysisSection.classList.add("hidden");
};

window.showCheckAnalysis = () => {
  checkAnalysisSection.classList.remove("hidden");
  startAttendanceForm.classList.add("hidden");

  const today = new Date().toISOString().split("T")[0];
  analysisDateInput.value = today;
  fetchAnalysis(today);
};

// ---------------------- START ANALYSIS ----------------------
window.startAnalysis = async () => {
  const course = document.getElementById("course").value;
  const semester = parseInt(document.getElementById("semester").value);
  const classroom = document.getElementById("classroom").value;
  const time = document.getElementById("time-slot").value;
  const date = new Date().toISOString().split("T")[0];

  try {
    await addDoc(collection(db, "analysisLogs"), {
      teacherId: currentUser.uid,
      course,
      semester,
      classroom,
      time,
      date,
      timestamp: serverTimestamp()
    });
    alert("Analysis started and saved!");
  } catch (error) {
    console.error("Error saving analysis:", error);
    alert("Failed to save analysis.");
  }
};

// ---------------------- FETCH ANALYSIS ----------------------
analysisDateInput?.addEventListener("change", (e) => {
  fetchAnalysis(e.target.value);
});

async function fetchAnalysis(date) {
  if (!currentUser) return;
  analysisResult.innerHTML = "<p>Loading...</p>";

  const q = query(
    collection(db, "analysisLogs"),
    where("teacherId", "==", currentUser.uid),
    where("date", "==", date)
  );

  try {
    const snapshot = await getDocs(q);
    if (snapshot.empty) {
      analysisResult.innerHTML = `<p>No records found for ${date}</p>`;
      return;
    }

    const logsByClass = {};

    snapshot.forEach(doc => {
      const data = doc.data();
      const classKey = `Classroom: ${data.classroom} | Semester: ${data.semester}`;
      if (!logsByClass[classKey]) logsByClass[classKey] = [];
      logsByClass[classKey].push(data);
    });

    let output = `<h4>Logs for ${date}</h4>`;
    for (let section in logsByClass) {
      output += `<strong>${section}</strong><ul>`;
      logsByClass[section].forEach(log => {
        output += `<li>${log.course} at ${log.time}</li>`;
      });
      output += `</ul>`;
    }

    analysisResult.innerHTML = output;
  } catch (error) {
    console.error("Error fetching logs:", error);
    analysisResult.innerHTML = `<p>Error fetching data.</p>`;
  }
}

// ---------------------- LOGOUT ----------------------
window.logout = async () => {
  await signOut(auth);
  window.location.href = "/login";
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

canvas.addEventListener("mousedown", e => { drawing = true; draw(e); });
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.lineTo(x, y);
  ctx.stroke();
}

document.getElementById("clear-btn").onclick = () => {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById("result").innerText = "";
};

document.getElementById("predict-btn").onclick = async () => {
  const dataUrl = canvas.toDataURL("image/png");
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl })
  });
  const result = await res.json();
  const top = result.top3[0];
  document.getElementById("result").innerHTML =
    `预测数字：<strong>${top.digit}</strong> （置信度 ${(top.prob * 100).toFixed(2)}%）<br>` +
    result.top3.map(t => `${t.digit}: ${(t.prob*100).toFixed(2)}%`).join("<br>");
};
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 12;           // 线条粗细
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "black";

// 允许多笔划，保留已有绘制内容
canvas.addEventListener("mousedown", e => {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

canvas.addEventListener("mousemove", e => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
});

let lastTime = 0;
canvas.addEventListener("mousemove", async e => {
  if (!drawing) return;
  const now = Date.now();
  if (now - lastTime > 500) { // 每0.5秒预测一次
    lastTime = now;
    const dataUrl = canvas.toDataURL("image/png");
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    });
    const result = await res.json();
    const top = result.top3[0];
    document.getElementById("result").innerHTML =
    `Prediction：<strong>${top.digit}</strong> （Confidence ${(top.prob * 100).toFixed(2)}%）<br>` +
    result.top3.map(t => `${t.digit}: ${(t.prob*100).toFixed(2)}%`).join("<br>");
  }
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.closePath();  // 不清空，只结束路径
});

canvas.addEventListener("mouseleave", () => {
  drawing = false;
  ctx.closePath();
});

// 清空按钮
document.getElementById("clear-btn").onclick = () => {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById("result").innerText = "";
};
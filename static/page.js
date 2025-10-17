const dynamicTextInH2 = document.querySelector("h2");
const h2Text = "Protect Your Content, Detect Deepfakes Instantly";

const typeEffectInH2 = () => {
    const words = h2Text.split(" "); 
    dynamicTextInH2.innerHTML = ""; 

    
    words.forEach((word, index) => {
        const span = document.createElement("span");
        span.textContent = word;
        dynamicTextInH2.appendChild(span);
    });
};

typeEffectInH2(); 
































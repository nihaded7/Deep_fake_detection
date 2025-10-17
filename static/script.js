const ctx = document.getElementById('accuracyChart').getContext('2d');
const accuracyChart = new Chart(ctx, {
    type: 'bar', 
    data: {
        labels: ['RESNET5O', 'CNN_LSTM'], 
        datasets: [{
            label: 'Accuracy (%)',
            data: [92, 78], 
            backgroundColor: [
                function() {
                    const gradient1 = ctx.createLinearGradient(0, 0, 0, 400);
                    gradient1.addColorStop(0, '#0d5f6f'); 
                    gradient1.addColorStop(1, '#1280a2'); 
                    return gradient1;
                },
                function() {
                    const gradient2 = ctx.createLinearGradient(0, 0, 0, 400);
                    gradient2.addColorStop(0, '#0d5f6f'); 
                    gradient2.addColorStop(1, '#1280a2'); 
                    return gradient2;
                }
            ],
            borderColor: ['#0b4d56', '#137b8c'],
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        animation: {
            duration: 1000,
            easing: 'easeOutBounce',
        },
        scales: {
            x: {
                display: true,
                grid: {
                    display: true
                },
                ticks: {
                    color: '#FFFFFF', 
                }
            },
            y: {
                display: false,
                grid: {
                    display: false
                },
                ticks: {
                    color: '#FFFFFF', // Couleur blanche pour les ticks de l'axe Y (si visible)
                }
            }
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    font: {
                        size: 14,
                        family: 'Arial',
                        weight: 'bold'
                    },
                    color: '#FFFFFF' 
                }
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(255, 255, 255, 0.8)', 
                titleColor: '#000000', 
                bodyColor: '#000000', 
                borderColor: '#ddd',
                borderWidth: 1
            },
            datalabels: {
                anchor: 'end',
                align: 'top',
                font: {
                    weight: 'bold',
                    size: 14
                },
                color: '#FFFFFF', 
                formatter: function(value) {
                    return value + '%'; 
                }
            }
        },
        elements: {
            bar: {
                borderRadius: 8,
            }
        }
    }
});


function startRecognition() {
    
    document.getElementById("decorative-image").style.display = "none";
    document.getElementById("videoElement").src = "/video_feed";
    document.getElementById("videoElement").classList.remove("hidden");
}

const dynamicText = document.querySelector("h1 span");
const words = ["Accurate", "Fast", "Reliable", "AI-Powered", "Real-Time"];

let wordIndex = 0;
let charIndex = 0;
let isDeleting = false;

const typeEffect = () => {
    const currentWord = words[wordIndex]; 
    const currentChar = currentWord.substring(0, charIndex); 
    dynamicText.textContent = currentChar; 
    dynamicText.classList.add("stop-blinking");

    
    if (!isDeleting && charIndex < currentWord.length) {
        charIndex++;
        setTimeout(typeEffect, 200); e
    } else if (isDeleting && charIndex > 0) {
        charIndex--;
        setTimeout(typeEffect, 100); 
    } else {
        
        isDeleting = !isDeleting;
        dynamicText.classList.remove("stop-blinking");
        wordIndex = !isDeleting ? (wordIndex + 1) % words.length : wordIndex; 
        setTimeout(typeEffect, 1200); 
    }
}

typeEffect();

const dynamicTextInSection = document.querySelector(".ex2 p");
const sentences = [
    "Our advanced AI models offer high accuracy in detecting deepfake content.",
    "Reliable and fast processing for real-time deepfake detection.",
    "Empowering users to verify the authenticity of digital media."
];

let sentenceIndex = 0; 

const typeEffectInSection = () => {
    const currentSentence = sentences[sentenceIndex]; 
    const words = currentSentence.split(" "); 
    dynamicTextInSection.innerHTML = ""; 

    
    words.forEach((word, index) => {
        const span = document.createElement("span");
        span.textContent = word;
        dynamicTextInSection.appendChild(span);
    });

    
    setTimeout(() => {
        sentenceIndex = (sentenceIndex + 1) % sentences.length; 
        typeEffectInSection(); 
    }, 4000); 

}

typeEffectInSection(); 

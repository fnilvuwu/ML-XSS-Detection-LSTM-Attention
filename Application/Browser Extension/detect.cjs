const swal = require("sweetalert2");
const urlWindow = window.location.href;
const INJECTION_KEYS = "value";
const detectXSSInject = () => {
    console.log("URL " + urlWindow);
    // Changet this to your API endpoint
    fetch("http://localhost:8000/predict", {
        method: "POST",
        mode: "cors",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            "text": urlWindow
        }),
    })
        .then((response) => {
            return response.json();
        })
        .then((text) => {
            console.log("testing " + text[INJECTION_KEYS]);
            const malProb = (text.probability * 100).toFixed(2);
            const conf = (text.confidence * 100).toFixed(2);
            if (text[INJECTION_KEYS] == 'Malicious') {
                swal.fire({
                    title: "Danger",
                    html: `The URL being accessed contains XSS <br /><br />
                       <b>Malicious Prob.</b> ${malProb}% <br />
                       <b>Confidence</b> ${conf}%`,
                    icon: "warning",
                    confirmButtonText: "Close",
                    allowOutsideClick: false,
                    allowEscapeKey: false,
                }).then((result) => {
                    if (result.isConfirmed) {
                        window.close();
                    }
                });

            } else {
                swal.fire({
                    title: "Safe",
                    html: `The URL being accessed is safe from XSS <br /><br />
                       <b>Malicious Prob.</b> ${malProb}% <br />
                       <b>Confidence</b> ${conf}%`,
                    icon: "success",
                    confirmButtonText: "Close",
                    allowOutsideClick: false,
                    allowEscapeKey: false,
                });
            }
        })
        .catch((error) => {
            console.log("Error Reponse " + error.status + "error response 2" + error.statusText);
        });
};

detectXSSInject();

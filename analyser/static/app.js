async function analysefiche() {
      const fileInput = document.getElementById('file');
      const outputDiv = document.getElementById('output');
      
      if (fileInput.files.length === 0) {
        outputDiv.innerHTML = "<p style='color: red;'>Veuillez sélectionner un fichier avant de lancer l'analyse.</p>";
        return;
      }

      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append('file', file); 
      outputDiv.innerHTML = "<p>Analyse en cours, veuillez patienter...</p>";

      try {
        const response = await fetch('http://localhost:8000/api/analyse', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          throw new Error(`Erreur serveur : ${response.status}`);
        }
        
        const data = await response.json();
        localStorage.setItem('analyseResultats', JSON.stringify(data));
        
        window.location.href = "resultats.html";

      } catch (error) {
        console.error("Erreur lors de la communication avec le serveur :", error);
        outputDiv.innerHTML = `<p style='color: red;'>Erreur : ${error.message}</p>`;
      }
    }
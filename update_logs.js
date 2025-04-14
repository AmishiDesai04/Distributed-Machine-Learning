function updateLogs() {
    fetch('log.txt')
        .then(response => response.text())
        .then(data => {
            document.getElementById('logs').innerText = data;
        });

    fetch('workers.txt')  // Fetch worker logs
        .then(response => response.text())
        .then(data => {
            document.getElementById('workerLogs').innerText = data;
        });
}

setInterval(updateLogs, 1000);
updateLogs();

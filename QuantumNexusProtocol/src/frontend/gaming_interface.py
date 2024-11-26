<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title >Gaming Interface - Quantum Nexus Protocol</title>
    <link rel="stylesheet" href="styles.css">
    <script src="app.js" defer></script>
</head>
<body>
    <header>
        <h1>Gaming Interface</h1>
    </header>
    <main>
        <h2>Welcome to the Gaming Zone</h2>
        <div id="game-list">
            <h3>Available Games</h3>
            <ul>
                <li>Game 1: <button onclick="playGame('game1')">Play Now</button></li>
                <li>Game 2: <button onclick="playGame('game2')">Play Now</button></li>
                <li>Game 3: <button onclick="playGame('game3')">Play Now</button></li>
                <!-- Additional games can be added here -->
            </ul>
        </div>
    </main>
    <footer>
        <p>&copy; 2023 Quantum Nexus Protocol. All rights reserved.</p>
    </footer>
    <script>
        function playGame(gameId) {
            // Logic to initiate the game based on the gameId
            alert(`Starting ${gameId}...`);
            // Redirect to the game interface or load the game
        }
    </script>
</body>
</html>

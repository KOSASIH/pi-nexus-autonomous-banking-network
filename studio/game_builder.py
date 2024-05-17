import json
import os
import random
from typing import Any, Dict, List, Union


class GameBuilder:
    """
    A high-tech game builder studio that generates a game configuration
    based on user input and randomizes certain aspects for added excitement!
    """

    def __init__(self, asset_path: str = "assets/"):
        """
        Initialize the GameBuilder object with an optional asset path.

        :param asset_path: The path to the assets directory. Defaults to "assets/".
        """
        self.asset_path = asset_path
        self.game_config: Dict[str, Union[str, List[str], Dict[str, Any]]] = {}

    def set_game_title(self, title: str) -> None:
        """
        Set the game title.

        :param title: The game title.
        """
        self.game_config["title"] = title

    def set_game_description(self, description: str) -> None:
        """
        Set the game description.

        :param description: The game description.
        """
        self.game_config["description"] = description

    def set_game_difficulty(self, difficulty: str) -> None:
        """
        Set the game difficulty (easy, medium, hard).

        :param difficulty: The game difficulty.
        """
        valid_difficulties = ["easy", "medium", "hard"]
        if difficulty not in valid_difficulties:
            raise ValueError(
                f"Invalid difficulty: {difficulty}. Valid difficulties are: {', '.join(valid_difficulties)}"
            )
        self.game_config["difficulty"] = difficulty

    def set_game_assets(self, assets: List[str]) -> None:
        """
        Set the game assets.

        :param assets: A list of asset file names.
        """
        self.game_config["assets"] = assets

    def generate_game_config(self) -> Dict[str, Union[str, List[str], Dict[str, Any]]]:
        """
        Generate the game configuration based on user input and randomization.

        :return: The game configuration.
        """
        self.game_config["randomized_aspect"] = random.choice(
            ["day/night cycle", "weather system", "dynamic soundtrack"]
        )
        self.game_config["game_id"] = os.urandom(16).hex()
        return self.game_config

    def save_game_config(self, file_name: str) -> None:
        """
        Save the game configuration to a JSON file.

        :param file_name: The name of the JSON file.
        """
        with open(os.path.join(self.asset_path, f"{file_name}.json"), "w") as f:
            json.dump(self.game_config, f, indent=4)

    def load_game_config(
        self, file_name: str
    ) -> Dict[str, Union[str, List[str], Dict[str, Any]]]:
        """
        Load a game configuration from a JSON file.

        :param file_name: The name of the JSON file.

        :return: The game configuration.
        """
        with open(os.path.join(self.asset_path, f"{file_name}.json"), "r") as f:
            return json.load(f)


# Example usage:
game_builder = GameBuilder()
game_builder.set_game_title("Epic Quest")
game_builder.set_game_description("Embark on a thrilling adventure!")
game_builder.set_game_difficulty("medium")
game_builder.set_game_assets(["character.png", "background.mp3"])

game_config = game_builder.generate_game_config()
print(game_config)

game_builder.save_game_config("epic_quest_config")
loaded_config = game_builder.load_game_config("epic_quest_config")
print(loaded_config)

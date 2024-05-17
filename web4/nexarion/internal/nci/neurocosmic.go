package nci

func Init() {}

func GetInput() string {
	var input string
	fmt.Print("Enter input: ")
	fmt.Scanln(&input)
	return input
}

func DisplayOutput(output string) {
	fmt.Println("Output:", output)
}

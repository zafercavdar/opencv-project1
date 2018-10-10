TEST_FILE = code/proj1_test_filtering.cpp
MAIN_FILE = code/project1.cpp
OUTPUT = proj1.out

clean:
	rm *.jpg
	rm $(OUTPUT)

compile-test:
	g++ -o $(OUTPUT) `pkg-config --cflags opencv` $(TEST_FILE) `pkg-config --libs opencv`

compile-main:
	g++ -o $(OUTPUT) `pkg-config --cflags opencv` $(MAIN_FILE) `pkg-config --libs opencv`

test: compile-test
	./$(OUTPUT)
	rm $(OUTPUT)

run: compile-main
	./$(OUTPUT)
	rm $(OUTPUT)

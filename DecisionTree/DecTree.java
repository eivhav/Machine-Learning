package ov4;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;

public class DecTree {
	
	static ArrayList<int[]> testInputList = new ArrayList<int[]>(); 
	static ArrayList<int[]> trainingInputList = new ArrayList<int[]>();
	static ArrayList<Node> tree = new ArrayList<Node>();
	static boolean randomImportance = true; 	// Specify here if you want to randomize to selection of attributes.  

	
	//The main method that starts the program. To run the code, please specify the location of the txt input files.  
	public static void main(String[] args) { 
		String testFile = "C:\\test.txt";  
		String trainingFile = "C:\\training.txt";  
		testInputList = input(testFile); 
		trainingInputList = input(trainingFile); 
		printTables(testInputList); 
		printTables(trainingInputList);
		Treebuilder(trainingInputList, randomImportance); 
		System.out.println("Accuracy: " + calculateAccuracy(tree.get(0), testInputList));  
	}
	
	
	
	// Builds a decision tree from scratch. Adds it to the Tree arrayList
	public static void Treebuilder(ArrayList<int[]> input, boolean rnd){
		
		int[] startFamily = new int[trainingInputList.get(0).length-1]; 
		int bestAtribute = getAtrByImportance(getAvailAtr(startFamily, 10), rnd); 
		
		Node startNode = new Node(null, 0, bestAtribute); 
		startNode.setFamilyList(startFamily); 
		tree.add(startNode); 
		createChildren(startNode, rnd, input); 
		printTree(startNode); 
	}
	
	
	// Generates children from a node, from the training datalist 
	public static void createChildren(Node parent, boolean rnd, ArrayList<int[]> input){
		int bestAtribute = getAtrByImportance(getAvailAtr(parent.getFamilyList(), parent.atrNumber), rnd); 
			Node child1 = new Node(parent, 1, bestAtribute); 
			Node child2 = new Node(parent, 2, bestAtribute); 
			tree.add(child1); tree.add(child2); 
			
			checkLeafNode(child1, input); 
			checkLeafNode(child2, input); 
			
			if(!child1.isIsleaf()){
				createChildren(child1, rnd, input); 
			}
			if(!child2.isIsleaf()){
				createChildren(child2, rnd, input); 
			} 
	}
	

	// Checks if the node is a leaf, and sets the final class value if so. 
	public static void checkLeafNode(Node child, ArrayList<int[]> input){
		int leafValue = leafNodeCalc(child.getFamilyList(), input); 
		if(leafValue == 1){
			child.setIsleaf(true);
			child.setFinalClass(1); 
		}
		else if(leafValue == 2){
			child.setIsleaf(true);
			child.setFinalClass(2);
		}
	}
	
	
	//Takes in a nodes family (attribute) history, and returns positions where there are available attributes  
	public static int[] getAvailAtr(int[] atr, int parentAtr){
		ArrayList<Integer> avAtr = new ArrayList<Integer>(); 
		for(int i = 0; i < atr.length ; i++){
			if(atr[i] == 0 && i != parentAtr){
				avAtr.add(i); 
			}
		}
		int[] returnArray = new int[avAtr.size()]; 
		for(int i = 0; i <avAtr.size(); i++){
			returnArray[i] = avAtr.get(i); 
		} 
		return returnArray; 
	}
	
	
	//Calculates and returns the attribute which has the most importance
	public static int getAtrByImportance(int[] availibleAtr, boolean rnd){
		if(availibleAtr.length > 0){
			int bestChoice = availibleAtr[0]; 
			for(int i = 1; i < availibleAtr.length; i++){ 
				if(importance(availibleAtr[i], rnd) >= importance(bestChoice, rnd)){
					bestChoice = availibleAtr[i]; 
				}
			}
			return bestChoice; 
		}
		else{ 
			return -1; 	
			}
		
	}
	
	
	// Calculates the importance for a given attribute. Random if rnd = true. Else by Entropy function 
	public static double importance(int artNumber, boolean rnd){
		Random rand = new Random();
		double rnd2 = rand.nextDouble();  
		if(rnd){
			return rnd2;} 
		else {
			int posClass = 1; 
			int negClass = 2; 
			double H = B( (calcEntr(posClass, -1, 0) / (calcEntr(posClass, -1, 0) + calcEntr(negClass, -1, 0)) ));  
			double P1N1 = (calcEntr(posClass, artNumber, 1) + calcEntr(negClass, artNumber, 1)); 
			double P2N2 = (calcEntr(posClass, artNumber, 2) + calcEntr(negClass, artNumber, 2)); 
			double PN = (calcEntr(posClass, -1, 0) + calcEntr(negClass, -1, 0)); 
			double p1 = calcEntr(posClass, artNumber, 1); 
			double p2 = calcEntr(posClass, artNumber, 2); 
			
			double Gain = H - ( ((P1N1 / PN) * B(p1 / P1N1)) + ((P2N2 / PN) * B(p2 / P2N2)) );     
			return Gain; 
		}
	}
	
	
	//Calculates the entropy(amount of objects) given class, attribute number and value  
	public static double calcEntr(int clas, int atrNum, int atrValue){ 
		double count = 0; 
		if(atrNum < 0)
			for(int i = 0; i < trainingInputList.size(); i++){
				if(trainingInputList.get(i)[trainingInputList.get(0).length-1] == clas){
					count++; 
				}
			}
		else 
			for(int i = 0; i < trainingInputList.size(); i++){
				if(trainingInputList.get(i)[trainingInputList.get(0).length-1] == clas && trainingInputList.get(i)[atrNum] == atrValue){
					count++; 
				}
			}
		return count; 
		
		
	}
	
	
	//Conversion to bits, as described in the textbook. 
	public static double B(double q){
		double qlogValue = Math.log(q) / Math.log(2); 
		double oneMinQlogValue = Math.log(1-q) / Math.log(2);
		double value = -((q*qlogValue) + ((1-q)* oneMinQlogValue)); 
		return value; 
	}
	

	//Calculates the accuracy degree for the decision tree, given an input list of objects
	public static double calculateAccuracy(Node rot, ArrayList<int[]> examples){
		double totalAccuracy = 0; 
		for(int i = 0; i < examples.size(); i ++){
			int[] object = examples.get(i);  
			Node child = rot;   
			for(int j = 0; j < object.length -1; j++){
				if(child.getChild1() != null && child.getChild1() != null){
					if(object[child.atrNumber] == 1){
						child = child.getChild1(); 
					}
					else if(object[child.atrNumber] == 2){
						child = child.getChild2(); 
					}
				}
				if(child.isIsleaf() && object[object.length-1] == child.getFinalClass()){
					totalAccuracy++;  
					break; 
				}
			}
		}	
		double accuracyDegree = totalAccuracy / ( (double)examples.size()); 
		return accuracyDegree; 		
	}
	
	
	//Determines if a node is a leaf or not. If the node is a leaf it also returns it's class, 1 or 2. If not, 0.  
	public static int leafNodeCalc(int[] testObject, ArrayList<int[]> examples){
		ArrayList<int[]> possibleObjects = new ArrayList<int[]>(); 
		for(int i = 0; i < examples.size(); i++){
			int[] object = examples.get(i);  
			boolean isTheSame = true; 
			
			for(int j = 0; j < object.length -1; j++){
				if(testObject[j] != 0 && testObject[j] != object[j]){
					isTheSame = false; 
					break; 				
				}	
			}
			if(isTheSame){
				possibleObjects.add(object); 
			}
		}
		
		int result = checkClass(possibleObjects); 
		return result;  
	}
	
	
	// Checks if all objects has the same class. 
	public static int checkClass(ArrayList<int[]> objects){
		int result = 0; 
		for(int i = 1; i < 3; i++){
			boolean isSameClas = true; 
			for(int j = 0; j < objects.size(); j++){
				if(objects.get(j)[objects.get(j).length-1] != i){
					isSameClas = false; 
					break; 
				}
			}
			if(isSameClas)	{
				result = i; 
			}
		}
		return result; 
	}
	
	
	
	
	
	
	// The input method.  
	public static ArrayList<int[]> input(String filepath){ 
		ArrayList<int[]> inputList = new ArrayList<int[]>(); 
		File file = new File((filepath)); 
		FileInputStream fis = null; 
		BufferedInputStream bis = null; 
		DataInputStream dis = null; 
		
		try { 
			fis = new FileInputStream(file);
			bis = new BufferedInputStream(fis); 
			dis = new DataInputStream(bis); 

			while (dis.available() != 0) { 
				
				ArrayList<Integer> object = new ArrayList<Integer>(); 
				String objectLine =  dis.readLine(); 
				
				for(int i = 0; i < objectLine.length(); i++){
					if(objectLine.charAt(i) == '1'){
						object.add(1); 
					}
					else if(objectLine.charAt(i) == '2'){
						object.add(2); 
					}
				}
				
				int[] intObj = new int[object.size()]; 
				for(int i = 0; i < object.size(); i++){
					intObj[i] = object.get(i); 	
				}
				inputList.add(intObj); 
			}
			fis.close(); 
			bis.close(); 
			dis.close(); 
		}
		catch (FileNotFoundException e) { 
		e.printStackTrace(); 
		}
		catch (IOException e) { 
		e.printStackTrace(); 
		} 
		return inputList; 
	} 
	

	
	
	
	// All methods below are print functions and support functions for these. 
	// Some of them was only used for debugging. 
	
	public static void printTree(Node startNode){
		System.out.println();
		System.out.println("no of nodes: " + tree.size());
		System.out.println("no of LeafNodes: " + getNoofLeafnodes(tree));

		ArrayList<Node> queue = new ArrayList<Node>(); 
		ArrayList<Node> printList = new ArrayList<Node>(); 
		queue.add(startNode); 
		printList.add(startNode); 
		
		ArrayList<Integer> pos = calcPrintPos(); 
		int count = 1; 
		
		while(queue.size() > 0 && count < pos.size()){
			Node node = queue.get(0); 
			queue.remove(0); 
			if(node != null){
				queue.add(node.getChild1()); 
				queue.add(node.getChild2());
				printList.add(node.getChild1());
				printList.add(node.getChild2());
			}
			else{ 
				queue.add(null); 
				queue.add(null); 
				printList.add(null);
				printList.add(null);
			}
			count++; count++;  
		}
		

		
		ArrayList<String> letterLines = getLines(true, printList, pos); 
		ArrayList<String> digitLines = getLines(false, printList, pos); 
		
		System.out.println(letterLines.get(0)); 
		for(int i = 1; i < letterLines.size(); i++){
			System.out.println(digitLines.get(i));
			System.out.println(letterLines.get(i)); 	 
		}	
	}

	public static ArrayList<String> getLines(boolean letter, ArrayList<Node> printList, ArrayList<Integer> pos){
		ArrayList<String> lines = new ArrayList<String>(); 
		int nodeCount = 0; 
		int range = 1; 
		for(int i = 0; i < 8; i++){
			String line = getSpaces(256);  
			
			for(int j = 0; j < range; j++){
				int posistion = pos.get(nodeCount);   
				String charact = "";  
				if(printList.get(nodeCount) != null && letter){
					charact = printList.get(nodeCount).getLetter();  
				}
				else if(printList.get(nodeCount) != null && !letter){
					charact = ""+ printList.get(nodeCount).getAtrValue();  
				}
				else{
					charact = " "; 
				}
				line = line.substring(0,posistion)+ charact + line.substring(posistion+1); 
				nodeCount++;  
			}
		lines.add(line); 
		range = range*2; 
		}
		return lines; 
	}
	
	
	
	public static int getNoofLeafnodes(ArrayList<Node> List){
		int count = 0; 
		for(int i = 0; i <List.size(); i++){
			if(List.get(i).isIsleaf())
				count++; 
		}
		return count; 
	}
	
	
	public static ArrayList<Integer> calcPrintPos(){
		ArrayList<Integer> pos = new ArrayList<Integer>(); 
		pos.add(128); 
		int value = 256; 
		for(int i = 1; i < 8; i++){
			int newValue = (int) (value / Math.pow(2, i)); 
			int stepValue = newValue /2; 
			int count = 0; 
			while((newValue*count) < value){
				  pos.add(stepValue + (newValue*count)); 
				  count++; 
			}		
		}
		return pos; 
	}
	
	public static String getSpaces(int size){
		String block = " "; 
		String space = ""; 
		for(int i = 0; i < size; i++){
			space = space + block; 
		}
		return space; 
	}
	
	public static void printTables(ArrayList<int[]> inputList){
		for(int i = 0; i < inputList.size(); i++){
			printArray(inputList.get(i)); 
		}
		System.out.println();
	}
	
	public static void printArray(int[] list){
		String line = ""; 
		for(int j = 0; j < list.length -1; j++){
			line = line + " " + list[j];  
		}
		line = line + " |" + list[list.length -1]; 
		System.out.println(line); 
	}

	public static void printArrayNOsep(int[] list){
		String line = ""; 
		for(int j = 0; j < list.length ; j++){
			line = line + " " + list[j];  
		}
		System.out.println(line); 
	}
	
	public static void printArrayList(ArrayList<Integer> list){
		String line = ""; 
		for(int j = 0; j < list.size() ; j++){
			line = line + " " + list.get(j);  
		}
		System.out.println(line); 
	}
	
	
	
	
	
	
	
	
	

}

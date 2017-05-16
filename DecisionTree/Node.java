package ov4;

import java.util.Arrays;

public class Node {
	
	boolean isleaf = false; 
	int[] familyList; 
	Node parentNode;  
	Node child1; 
	Node child2; 
	int finalClass; 
	int atrNumber; 
	int atrValue; 
	int bestAtribute; 
	
	
	public Node(Node parentNode, int atrValue, int atrNumber){
		if(parentNode != null){ 
			this.parentNode = parentNode; 		
			familyList = Arrays.copyOf(parentNode.getFamilyList(), parentNode.getFamilyList().length);  
			familyList[parentNode.atrNumber] = atrValue;  
			this.atrNumber = atrNumber; 
			this.atrValue = atrValue; 
			if(atrValue == 1){
				parentNode.setChild1(this); }
			else if(atrValue == 2){
				parentNode.setChild2(this);
			}
		}
		else
			this.atrNumber = atrNumber; 
	
	}
	
	
	public String getLetter(){
		String out = "";  
		if(isleaf){
			if(finalClass == 1)
			out = "X" ; 
			else if(finalClass == 2)
				out = "Y"; 
		}
		else{
			if(atrNumber == 0){
				out = "A"; }
			else if(atrNumber == 1){ 
				out = "B"; }
			else if(atrNumber == 2){ 
				out = "C"; }
			else if(atrNumber == 3){ 
				out = "D"; }
			else if(atrNumber == 4){ 
				out = "E"; }
			else if(atrNumber == 5){ 
				out = "F"; }
			else if(atrNumber == 6){ 
				out = "G"; }
		}
		return out; 
		
	}
	
	public int getAtrValue(){
		return atrValue; 
	}
	
	
	public int getTreeLevel(){
		int count = 0; 
		for(int i = 0; i < familyList.length; i++){
			if(familyList[i] != 0){
				count++; 
			}
		}
		return count; 
	}

	public int getFinalClass() {
		return finalClass; 
	}
	
	public void setFinalClass(int clas) {
		this.finalClass = clas;  
	}

	public boolean isIsleaf() {
		return isleaf;
	}

	public void setIsleaf(boolean isleaf) {
		this.isleaf = isleaf;
	}

	public int[] getFamilyList() {
		return familyList;
	}

	public void setFamilyList(int[] familyList) {
		this.familyList = familyList;
	}

	public Node getParentNode1() {
		return parentNode;
	}

	public void setParentNode(Node parentNode1) {
		this.parentNode = parentNode1;
	}


	public Node getChild1() {
		return child1;
	}


	public void setChild1(Node child1) {
		this.child1 = child1;
	}


	public Node getChild2() {
		return child2;
	}


	public void setChild2(Node child2) {
		this.child2 = child2;
	}
	
	



	
}

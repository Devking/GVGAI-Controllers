/*********************************************************************
** Code written by Wells Lucas Santo
** This code was written as part of the CS9223 course at NYU.
** If you wish to reproduce this code in any way, please give credit.
**********************************************************************/

// Intention: To develop a controller that used minimal to none domain specific knowledge
// in making its decisions. This is a simple **MCTS** controller that does random playouts to
// a certain depth (not always to end-game) and assigns value to that state based on the
// *score* at that state. (Or, if the game did finish, give value to winning and losing.)

// Conclusion: The controller is extremely good at games like Space Invaders, where enemies
// are far away, but is afraid to go into situations where death would be likely (such as
// when going close to enemy NPCs). The controller also has no motivation to move around
// and explore the grid in a game where score does not easily increase (since future steps may
// not guarantee higher scores or win/loss). One fix is to implement something to check if 
// the controller has stayed in one location on the map for too long and penalize for that.

// However, this does not quite work--so I made the controller somewhat "aggressive" in the
// sense that it would value positions closer to NPCs but not moving objects, and it would value
// states with less NPCs in them. (Note that for games where you don't try to beat NPCs, this
// heuristic shouldn't do anything.) I also lowered the amount of "bad reward" of deaths,
// to encourage participation in the game (instead of just running away).

// How it works:
// This is a simple MCTS controller that uses the UCT function to determine which child to
// further explore in the tree. This MCTS algorithm does not implement any of the further
// improvements discussed in class -- it merely does the standard selection, expansion,
// simulation, and backpropagation steps as described exactly by the standard vanilla MCTS
// algorithm. However, in evaluating the final state, there is room to add heuristics.

// Note UCT function:
// (reward of child/ number of visits to child) + (weight) sqrt ( (2*ln(number of root visits)) / (number of child visits) )

// Note: This is an 'anytime' algorithm--so even if the 'tree search' stops early, we still know 
// what the best next move to make is.

package Savague;

// basic imports to allow the controller to work
import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import tools.ElapsedCpuTimer;
// import needed for getting Types.ACTIONS and Types.WINNER
import ontology.Types;
// import needed for dealing with locations on the grid
import tools.Vector2d;
// import needed for generating random numbers
import java.util.Random;
// import needed for dealing with some java functionality
import java.awt.*;
import java.util.*;
import java.util.ArrayList;


public class Agent extends AbstractPlayer {

	final int expansionDepth = 25;
	Node root;
	int[][] positionCount;

	public class Node {

		Node (StateObservation s, Node par) {
			thisState = s;
			totalReward = 0;
			visitCount = 0;
			parent = par;
			children = new Node[ thisState.getAvailableActions().size() ];
			if (par != null)
				depth = par.depth+1;
			else
				depth = 0;
		}

		public StateObservation thisState;
		public double totalReward;
		public int visitCount;
		public Node parent;
		public Node[] children;
		public int depth;
	}

	// constructor, where the controller is first created to play the entire game
	public Agent(StateObservation states, ElapsedCpuTimer elapsedTime) {
		// initialize the positionCount to all 0's and the size of the entire grid
		//ArrayList<Observation>[][] gridSpace = states.getObservationGrid();
		int x = 1000;
		int y = 1000;
		//System.out.println("x: " + x + " y: " + y);
		positionCount = new int[x][y];
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				positionCount[i][j] = 0;
			}
		}
	}

	// T - initialize the root and begin the UCT search
	public int runMCTS (StateObservation origState, ElapsedCpuTimer origTime) {
		root = new Node (origState, null);
		//System.out.println(root.children.length);
		while (origTime.remainingTimeMillis() > 3.0) {
			// expanded should be a child of 'root'
			Node expanded = treePolicy(root);
			// assignReward should reflect the value of 'expanded'
			// that is, for the 'bottommost' node we have chosen to expand,
			// we will calculate how 'good' the state it represents is
			double valueChange = assignReward(expanded, origTime);
			//System.out.println("value:" + valueChange);
			//System.out.println("----------------------");
			// we remember how 'good' that state is, and propagate that
			// reward + visit increment up the tree
			backProp(expanded, valueChange);
			//System.out.println("visit: " + root.visitCount);
		}
		//System.out.println("remaining time:" + origTime.remainingTimeMillis());
		return mostRewardChild(root);
	}

	// T - we will choose to expand the root repeatedly until we get to a leaf
	// note that we DO NOT evaluate newly created leaves here
	public Node treePolicy ( Node roNode ) {
		// while we haven't hit the very bottom of our tree, we will
		// continue to expand at this node
		Node thisNode = roNode;
		while (!thisNode.thisState.isGameOver() && expansionDepth > thisNode.depth ) {
			// if this particular node has not been fully expanded, we must
			// visit all of its children at least once (gives diversity to our tree)
			// "wouldn't this just expand every single child of the tree?""
			// no, because once we explore the entire first row, we will only
			// pick the child that is best and explore the rest of that child
			for (int i = 0; i < thisNode.children.length; i++) {
				if (thisNode.children[i] == null) {
					return expandTree(thisNode, i);
				}
			}
			// otherwise, we will replace the 'exploredNode' with the 
			// child that reaps the best reward (pick best descendent to update in value)
			// second parameter determines how much we value diversity in exploration
			thisNode = bestChild(thisNode, 0.1);
			//System.out.println("Depth:" + roNode.depth);
		}
		// after we've explored the best route to the bottom of the tree,
		// we will return the node that corresponds to the very bottom of the tree
		// in this expansion
		return thisNode;
	}

	// D - add new unexplored node to the root node that is advanced by one action
	// we will not evaluate how good the action is here; we just make the move
	public Node expandTree ( Node roNode, int childNo ) {
		// System.out.println(childNo + " expansion!");
		// expanding action corresponding to 'childNo' in 'roNode'
		StateObservation childState = roNode.thisState.copy();
		//System.out.println( roNode.thisState.getAvailableActions().get(childNo) );
		childState.advance( roNode.thisState.getAvailableActions().get(childNo) );
		// add that child to the root node
		roNode.children[childNo] = new Node (childState, roNode);
		// return child node
		return roNode.children[childNo];
	}

	// NN - in normal considerations, a random playout until the end of the game
	// is done... but this may not be desired here; instead, just assign an award
	// for this node based on some heuristic
	// NOTE: this 'reward' will be propagated up the ENTIRE tree!
	// be sure to make 'win' and 'lose' rewards equal so that parent nodes
	// can be 'balanced' out by wins and loses
	public double assignReward ( Node baseNode, ElapsedCpuTimer origTime) {
		StateObservation finalState = baseNode.thisState.copy();
		int finalDepth = baseNode.depth;
		Random random = new Random();
		// make sure that for any of the nodes we are exploring, we
		// go as deep as we can using random playouts
		while (finalDepth < expansionDepth && origTime.remainingTimeMillis() > 5.0 && !finalState.isGameOver()) {
			int actionNo = random.nextInt(finalState.getAvailableActions().size());
			finalState.advance( finalState.getAvailableActions().get(actionNo) );
			finalDepth++;
		}
		double stateVal = 0;
		if (finalState.getGameWinner() == Types.WINNER.PLAYER_WINS) { stateVal += 100000; }
		// give lower value on losing?
		if (finalState.getGameWinner() == Types.WINNER.PLAYER_LOSES) { stateVal -= 5000; }
		stateVal += finalState.getGameScore();
		// the player keeps freezing in place... how to adjust for this?
		// count how many times the player has been at this specific spot in the game, and penalize!
		// but doesn't quite work out..
		Vector2d myPosition = finalState.getAvatarPosition();
		//System.out.println(myPosition.x + " y: " + myPosition.y);
		//int x = (int) myPosition.x;
		//int y = (int) myPosition.y;
		//positionCount[x][y]++;
		//stateVal -= positionCount[x][y];
		// perhaps move away from moving objects?
		/*
		ArrayList<Observation>[] movingObjects = finalState.getMovablePositions(myPosition);
		if (movingObjects != null && movingObjects.length > 0 && movingObjects[0].size() > 0) {
			Vector2d closestObject = movingObjects[0].get(0).reference;
			double distanceToObj = myPosition.dist(closestObject);
			stateVal -= distanceToObj;
		}*/
		// also encourage movement towards NPCs?
		ArrayList<Observation>[] movingNPCs = finalState.getNPCPositions(myPosition);
		if (movingNPCs != null && movingNPCs.length > 0 && movingNPCs[0].size() > 0) {
			int totalNPCs = 0;
			for (int i = 0; i < movingNPCs.length; i++) { totalNPCs += movingNPCs[i].size(); }
			// if we can have less NPCs, then this is good as well?
			stateVal -= totalNPCs*2;
			Vector2d closestNPC = movingNPCs[0].get(0).reference;
			double distanceToNPC = myPosition.dist(closestNPC);
			stateVal -= distanceToNPC;
		}
		return stateVal;
	}

	// D - propagate number of visits and reward up the tree from the explored node
	public void backProp (Node baseNode, double value) {
		Node no = baseNode;
		while (no != null) {
			no.visitCount++;
			no.totalReward += value;
			//System.out.println("Rewarddddd: " + no.totalReward);
			no = no.parent;
		}
	}

	// NN - given 'someNode', use the UCT function to see which of its
	// children reaps the best reward -- this always checks ALL
	// children, because this can only be called after all children have been explored once
	public Node bestChild ( Node someNode, double weight ) {
		// calculate best child given each child's reward and exploration
		// if multiple children have the same reward, explore the first one
		// (reward of child/ number of visits to child) + (weight) sqrt ( (2*ln(number of root visits)) / (number of child visits) )
		int bestindex = someNode.children.length-1;
		double bestValue = -Double.MAX_VALUE;
		int totalVisits = someNode.visitCount;
		for (int i = 0; i < someNode.children.length; i++) {
			Node child = someNode.children[i];
			double reward = child.totalReward;
			int childVisits = child.visitCount;
			// the below equation is based on the UCT equation provided in the course
			double thisValue = (reward)/(childVisits) + weight * Math.sqrt((2*Math.log(totalVisits))/childVisits);
			if (thisValue > bestValue) {
				bestValue = thisValue;
				bestindex = i;
			}
		}
		return someNode.children[bestindex];
	}

	// D - this will take the root node and go through, looking at the index
	// of the children and pick the child with the highest reward
	// ...then return the index of that child, which corresponds to the next action!
	public int mostRewardChild ( Node roNode ) {
		double bestReward = -Double.MAX_VALUE;
		int bestChildNo = roNode.children.length-1;
		for (int i = 0; i < roNode.children.length; i++) {
			if (roNode.children[i] != null) {
				//System.out.println("reward: " + roNode.children[i].totalReward);
				if (roNode.children[i].totalReward > bestReward) {
					bestReward = roNode.children[i].totalReward;
					bestChildNo = i;
				}
			}
		}
		return bestChildNo;
	}

	// D - at every 'state' the controller must make an action -- this method is thus called
	// this MUST return in 40 ms to correctly perform an action -- this is given by 'origTime'
	public Types.ACTIONS act(StateObservation origState, ElapsedCpuTimer origTime) {
		// startMCTS will initialize the root and start the search, returning an int
		// that relates to the best action to take from this state
		return origState.getAvailableActions().get( runMCTS(origState, origTime) );
	}

}
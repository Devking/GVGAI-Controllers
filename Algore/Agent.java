/*********************************************************************
** Code written by Wells Lucas Santo
** This code was written as part of the CS9223 course at NYU.
** If you wish to reproduce this code in any way, please give credit.
**********************************************************************/

// Neuro-evolution algorithm

// Note: Sadly, I do not retain the weighting for neural nets between
// games--that is, at the start of any game, the population will do badly
// because none of its weights have been evolved... but as the game progresses
// the weighting will become better and the player will make better moves!

// have a population of neural nets, with different weights on:
// input: grid information -- distance to closest NPC, number of NPC, score, 
// distance to closest portal, distance to closest moving object, distance to
// closest resource, number of resources, gameState (8 inputs)
// output: number of actions that can be performed (outputs = # of actions)

// take the max of the outputs and perform that action
// have each of the neural nets take 'n' iterations into the game
// pick the 'mu' neural nets that do the best and mutate them
// eventually we expect to have a neural net with weights that play
// the game well

// this algorithm does not attempt to update the neural nets themselves
// once time is up, or we have gone through enough generations,
// we will perform the action of the "best" neural net

// we will be sure to store the population of neural nets that we already have
// so we can pick up where we left off at each action in the game

// TUNABLE THINGS:
// "mutateStep": How much should weights differ between mutations?
// State Heuristics: What's a good measure of any state?
// Mu, Lambda, Population Size, and Generation Length

// Potential Issues: The inputs to the neural nets are specific states
// However, the states of the game will change over time -- there's no
// guarantee that a 'good move' for an earlier state will be a 'good move'
// for a later state. This may lead to 'risky' behavior that is tied too
// strongly to tropisms of the past.

package Algore;

// basic imports to allow the controller to work
import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import tools.ElapsedCpuTimer;
// import needed for getting Types.ACTIONS and Types.WINNER
import ontology.Types;
// import needed for dealing with locations on the grid
import tools.Vector2d;
// import needed for dealing with some java functionality
import java.awt.*;
import java.util.*;
import java.util.ArrayList;
// import needed for random number generation
import java.util.Random;

public class Agent extends AbstractPlayer {

	final static int popSize = 6;
	final static int muSize = 2;
	final static int lamSize = popSize - muSize;
	final static int noGenerations = 4;
	final static double mutateStep = 1.5;
	static Random rng = new Random();
	static NeuralNet[] population = new NeuralNet[popSize];
	static StateObservation[] popCopies = new StateObservation[popSize];

	final int origResourceNo;
	final int origNPCNo;

	public static class NeuralNet implements Comparable<NeuralNet> {
		// input layer -> hidden layer -> output layer
		// these are the values actually held at each node
		double[] inputs;
		double[] outputs;
		double[] hiddenLayer;
		// these are the weights of the connections from each
		// node at the first layer to each node at the second layer
		double[][] inputToHidden;
		double[][] hiddenToOutput;
		// this is the 'value' of this neural net to sort by
		double score;
		// keep track of the first action that this net did, for this step
		int firstAction;

		// to allow for comparisons of neural nets
		public int compareTo(NeuralNet another) { return (this.score > another.score) ? 1 : -1; }

		// for creating a neural net from scratch -- this is the Biblical Adam
		public NeuralNet(int noInput, int noHidden, int noOutput) {
			inputs = new double[noInput];
			hiddenLayer = new double[noHidden];
			outputs = new double[noOutput];
			inputToHidden = new double[noInput][noHidden];
			hiddenToOutput = new double[noHidden][noOutput];
			score = 0.0;
		}

		// for creating a copied but mutated neural net
		public NeuralNet(NeuralNet parent) {
			inputs = new double[parent.inputs.length];
			hiddenLayer = new double[parent.hiddenLayer.length];
			outputs = new double[parent.outputs.length];
			inputToHidden = new double[parent.inputToHidden.length][parent.inputToHidden[0].length];
			hiddenToOutput = new double[parent.hiddenToOutput.length][parent.hiddenToOutput[0].length];
			// copy weights from the parent, but with slight mutations
			for (int i = 0; i < inputToHidden.length; i++)
				for (int j = 0; j < inputToHidden[i].length; j++)
					inputToHidden[i][j] += (rng.nextDouble()-0.5) * mutateStep;
			for (int i = 0; i < hiddenToOutput.length; i++)
				for (int j = 0; j < hiddenToOutput[i].length; j++)
					hiddenToOutput[i][j] += (rng.nextDouble()-0.5) * mutateStep;
			score = 0.0;
		}

		// take in array corresponding to input values
		// output array corresponding to output values
		public double[] fullExcitation (double[] firstInputs) {
			// Check that the input size is correct
			if (firstInputs.length != inputs.length) {
				System.out.println("Mismatch in input length!");
				return firstInputs;
			}
			// Copy the input values into the first layer of neurons
			for (int i = 0; i < firstInputs.length; i++) inputs[i] = firstInputs[i]; 
			// clear the values at the hidden and output layer
			resetLayer(hiddenLayer);
			resetLayer(outputs);
			// by using the values from the input layer
			// as well as the learned weights, we will find the values
			// of the hidden layer by taking in appropriate inputs
			propagateLayer(inputs, hiddenLayer, inputToHidden);
			// once the "input" values are received, run sigmoid to update
			// all of the values in the hidden layer
			sigmoid(hiddenLayer);
			// by using the values from the hidden layer
			// as well as the learned weights, we will find the values
			// of the output layer by taking in inputs from hidden layer
			propagateLayer(hiddenLayer, outputs, hiddenToOutput);
			// once all the hidden layer values are received, run sigmoid
			// to update all of the values in the output
			sigmoid(outputs);
			return outputs;
		}

		// Clear all the stored values at a specific layer
		public void resetLayer (double[] array) { for (int i = 0; i < array.length; i++) array[i] = 0; }

		// Using all of the inputs to a layer, as well as the weights given to each input
		// for each neuron, compute the linear combination of values for each neuron
		public void propagateLayer (double[] layerOne, double[] layerTwo, double[][] weights) {
			for (int i = 0; i < layerOne.length; i++)
				for (int j = 0; j < layerTwo.length; j++) { layerTwo[j] += layerOne[i] * weights[i][j]; }
		}

		// Replace the value at that neuron with the value after evaluation by a sigmoid
		public void sigmoid (double[] layerValues) {
			for (int i = 0; i < layerValues.length; i++) 
				layerValues[i] = 1 / ( 1 + Math.pow(Math.E,layerValues[i]) );
		}
	}

	// given this neural net, (and its inputs), what action shall it perform?
	public Types.ACTIONS chooseAction (StateObservation origState, NeuralNet thisNet) {
		int bestAction = 0;	// note, we might accidentally give preference to action 0 here
		double actionWeight = thisNet.outputs[0];
		// pick the output with the highest value -- this corresponds to the action we will take
		for (int i = 0; i < thisNet.outputs.length; i++) {
			if (actionWeight < thisNet.outputs[i]) {
				bestAction = i;
				actionWeight = thisNet.outputs[i];
			}
		}
		return origState.getAvailableActions().get(bestAction);
	}

	// constructor, where the controller is first created to play the entire game
	public Agent(StateObservation states, ElapsedCpuTimer elapsedTime) {
		// do all initializations here
		// get number of resources at the start of the game
		Vector2d myPosition = states.getAvatarPosition();
		ArrayList<Observation>[] resources = states.getResourcesPositions(myPosition);
		int noResource = 0;
		if (resources != null)
			for (int i = 0; i < resources.length; i++) { noResource += resources[i].size(); }
		origResourceNo = noResource;
		//System.out.println("Number of Resources: " + noResource);
		// get number of NPCs at the start of the game
		ArrayList<Observation>[] npcs = states.getNPCPositions(myPosition);
		int noNPC = 0;
		if (npcs != null)
			for (int i = 0; i < npcs.length; i++) { noNPC += npcs[i].size(); }
		origNPCNo = noNPC;
		//System.out.println("Number of NPCs: " + noNPC);
		// create the first neural net
		// 7 inputs, 5 hidden neurons, and outputs = # of actions
		int noActions = states.getAvailableActions().size();
		NeuralNet adam = new NeuralNet(8, 5, noActions);
		population[0] = adam;
		// create the other neural nets (they will all have mutated weights)
		for (int i = 1; i < popSize; i++)
			population[i] = new NeuralNet(population[i-1]);
		//System.out.println("Done!");
	}

	// should return an array of doubles that we want to feed as an input to our neural net
	// (0) distance to closest NPC, (1) number of NPC, (2) score, (3) distance to closest portal, 
	// (4) distance to closest moving object, (5) distance to closest resource, 
	// (6) number of resources, (7) gameState (8 inputs)
	public double[] stateValue(StateObservation thisState) {
		double[] inputs = new double[8];

		Vector2d myPosition = thisState.getAvatarPosition();

		// 0. Distance to closest NPC
		// 1. Number of NPC
		ArrayList<Observation>[] npcPositions = thisState.getNPCPositions(myPosition);
		int noNPC = 0;
		double distanceNPC = 0;
		if (npcPositions != null) {
			for (int i = 0; i < npcPositions.length; i++) { noNPC += npcPositions[i].size(); }
			if (npcPositions[0].size() != 0) {
				Vector2d closestNPC = npcPositions[0].get(0).position;
				distanceNPC = myPosition.dist(closestNPC);
			}
		}
		inputs[0] = distanceNPC;
		inputs[1] = noNPC;

		// 2. Score input
		inputs[2] = thisState.getGameScore();

		// 3. Distance to closest portal
		ArrayList<Observation>[] portalPositions = thisState.getPortalsPositions(myPosition);
		double distancePortal = 0;
		if (portalPositions != null) {
			if (portalPositions[0].size() != 0) {
				Vector2d closestPortal = portalPositions[0].get(0).position;
				distancePortal = myPosition.dist(closestPortal);
			}
		}
		inputs[3] = distancePortal;

		// 4. Distance to closest moving object
		ArrayList<Observation>[] movePositions = thisState.getMovablePositions(myPosition);
		double distanceMove = 0;
		if (movePositions != null) {
			if (movePositions[0].size() != 0) {
				Vector2d closestMove = movePositions[0].get(0).position;
				distanceMove = myPosition.dist(closestMove);
			}
		}
		inputs[4] = distanceMove;

		// 5. Distance to closest resource
		// 6. Number of resources
		ArrayList<Observation>[] resourcesPositions = thisState.getResourcesPositions(myPosition);
		int noResource = 0;
		double distanceResource = 0;
		if (resourcesPositions != null) {
			for (int i = 0; i < resourcesPositions .length; i++) { noResource += resourcesPositions [i].size(); }
			if (resourcesPositions [0].size() != 0) {
				Vector2d closestResource = resourcesPositions[0].get(0).position;
				distanceResource = myPosition.dist(closestResource);
			}
		}
		inputs[5] = distanceResource;
		inputs[6] = noResource;

		// 7. Game State input
		if (thisState.getGameWinner() == Types.WINNER.PLAYER_WINS) { inputs[7] = 9999; }
		else if (thisState.getGameWinner() == Types.WINNER.PLAYER_LOSES) { inputs[7] = -9999; }
		else { inputs[7] = 0; }

		return inputs;
	}

	// we can use some sort of heuristics to calculate how "good" this state is
	// right now, I only consider the score, win/loss states, # of npcs, and # of resources
	public double evaluateState (StateObservation origState) {

		double value = 0;

		// give values to win/loss states
		if (origState.getGameWinner() == Types.WINNER.PLAYER_WINS) { value += 9999; }
		else if (origState.getGameWinner() == Types.WINNER.PLAYER_LOSES) { value -= 9999; }

		// give value to score of the game
		value += origState.getGameScore();

		Vector2d myPosition = origState.getAvatarPosition();

		// give value to number of NPCs currently existing
		// 1 - (NPCs now)/(NPCs beginning)
		if (origResourceNo != 0) {
			ArrayList<Observation>[] resources = origState.getResourcesPositions(myPosition);
			int noResource = 0;
			if (resources != null)
				for (int i = 0; i < resources.length; i++) { noResource += resources[i].size(); }
			double resourceSpec = 1 - noResource/origResourceNo; // the less resources, the higher the number
			value += resourceSpec * 2; // can multiply this to weight it
		}

		// give value to number of resources remaining
		// 1 - (resources now)/(resources beginning)
		if (origNPCNo != 0) {
			ArrayList<Observation>[] npcs = origState.getNPCPositions(myPosition);
			int noNPC = 0;
			if (npcs != null)
				for (int i = 0; i < npcs.length; i++) { noNPC += npcs[i].size(); }
			double npcSpec = 1 - noNPC/origNPCNo; // the less NPCs, the higher the number
			value += npcSpec * 0.5; // can multiply this to weight it
		}

		return 0;
	}

	// at every 'state' the controller must make an action -- this method is thus called
	// this MUST return in 40 ms to correctly perform an action -- this is given by 'origTime'
	public Types.ACTIONS act(StateObservation origState, ElapsedCpuTimer origTime) {

		// while time still permits
		while (origTime.remainingTimeMillis() > 1.0) {

			// make 1 copy of the original state for each neural net to explore
			// in future iterations, this serves to refresh the state cleanly
			for (int i = 0; i < popSize; i++)
				popCopies[i] = origState.copy();

			// for each neural net
			for (int i = 0; i < popSize; i++) {
				// for the number of generations we wish to iterate:
				for (int j = 0; j < noGenerations; j++) {
					if (popCopies[i].isGameOver()) break;
					// get information about the state associated with this net
					double[] inputs = stateValue(popCopies[i]);
					// use state information as inputs to this neural net
					// and then update the output values of this neural net
					population[i].fullExcitation(inputs);
					// have the neural net pick an action to perform based on outputs
					// and advance the state based on the action selected
					popCopies[i].advance( chooseAction(popCopies[i], population[i]) );
					if (origTime.remainingTimeMillis() < 1.0) break;
				}
				// evaluate the score of this neural net
				population[i].score = evaluateState(popCopies[i]);
				if (origTime.remainingTimeMillis() < 1.0) break;
			}

			//System.out.println("One universe to rule them all");

			// sort all neural nets by their score (with the best at the end)
			Arrays.sort(population);
			// replace 'lamSize' neural nets with mutations from either of the best neural nets
			for (int i = 0; i < lamSize; i++) {
				int offset = rng.nextInt(muSize-1)+1;
				population[i] = new NeuralNet(population[popSize-offset]);
			}
		}

		// we will just have the "best" neural net do the move it wants to do for the original state
		return chooseAction(origState, population[popSize-1]);
	}

}
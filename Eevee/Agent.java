/*********************************************************************
** Code written by Wells Lucas Santo
** This code was written as part of the CS9223 course at NYU.
** If you wish to reproduce this code in any way, please give credit.
**********************************************************************/

// Intention: To experiment with combining the concept of **evolution** with an unbalanced
// tree search with bias towards future moves (found through random playouts) that are "better".

// Conclusion: Not a very strong algorithm--depends too heavily on how good the heuristics
// used in state evaluation can tell what a 'better' state is. The search is also too shallow
// and assumes too much determinism in the movements of the NPCs. Picking a 'random' next move
// to perform is also counter to searching a tree thoroughly.

// How it works:
// A *very* simple 'tree search' controller motivated by evolutionary algorithms to pick
// which future nodes to explore. This is motivated by a basic mu+lambda evolutionary algorithm,
// where the best 'mu' states are explored, and the remaining 'lambda' states are replaced.
// We repeat this process for a number of generations, and at that point, select the first action of
// the earliest ancestor of the individual in the population who has the best state. (That is, pick the
// move that can get us towards the state of this "best" individual.)
// Will use some heuristics to do state evaluation in order to pick 'best' in the population.

// Note: This is an 'anytime' algorithm--so even if the 'tree search' stops early, we still know 
// what the best next move to make is.

package Eevee;

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

	int actionNo = 0;
	final int populationSize = 6;
	final int muSize = 2;
	final int lamSize = populationSize - muSize;
	final int totalGenerations = 30;
	Random rng = new Random();

	public class StateTuple implements Comparable<StateTuple> {
		public StateTuple(int x, double y) {
			stateno = x;
			statescore = y;
		}
		public int compareTo(StateTuple another) {
			return (this.statescore > another.statescore) ? 1 : -1;
		}
		public int stateno;
		public double statescore;
	}

	public class StateAndAncestor {
		public StateAndAncestor(StateObservation me, int ances) {
			myState = me;
			ancesNo = ances;
		}
		public int ancesNo;
		public StateObservation myState;
	}

	// constructor, where the controller is first created to play the entire game
	public Agent(StateObservation states, ElapsedCpuTimer elapsedTime) {
		// do all initializations here
	}

	// evaluate a specific state based on some heuristics
	// the current heuristics: value victory and higher score most, and attempt to move towards resources (if they exist)
	// if no resources exist, move towards portals (if they exist)
	// npc's in different games hold different meanings--some you want to get close to, some you don't
	// this works badly in games that do not utilize 'score'
	public double stateEval ( StateObservation someState ) {

		double stateVal = 0;

		double score = someState.getGameScore();
		Vector2d myPosition = someState.getAvatarPosition();
		ArrayList<Observation>[] npcPositions = someState.getNPCPositions(myPosition);
		ArrayList<Observation>[] portalPositions = someState.getPortalsPositions(myPosition);
		ArrayList<Observation>[] resourcesPositions = someState.getResourcesPositions(myPosition);

		if (someState.getGameWinner() == Types.WINNER.PLAYER_WINS) { return 999999999; }
		if (someState.getGameWinner() == Types.WINNER.PLAYER_LOSES) { return -99999999; }

		// better value for higher scores
		stateVal += score * 100;

		// better value if closer to closest resource of each type
		// but even better if less resources (means we picked it up)
		int noResources = 0;
		if (resourcesPositions != null) {
			for (int i = 0; i < resourcesPositions.length; i++) {
				noResources += resourcesPositions[i].size();
				if (resourcesPositions[i].size() > 0) {
					Vector2d closestResPos = resourcesPositions[i].get(0).position;
					double distToResource = myPosition.dist(closestResPos);
					// the farther away it is, the worst the stateVal will be
					stateVal -= distToResource*5;
				}
			}
		}
		stateVal -= noResources * 100;

		// better value if closer to closest portal of each type
		// what if there is a wall between us and the portal? --> in 'zelda' this is why we die
		if (portalPositions != null) {
			for (int i = 0; i < portalPositions.length; i++) {
				if (portalPositions[i].size() > 0) {
					Vector2d closestPorPos = portalPositions[i].get(0).position;
					double distToPortal = myPosition.dist(closestPorPos);
					// the farther away it is, the worst the stateVal will be
					stateVal -= distToPortal / 5;
				}
			}
		}

		// better value if less NPCs
		int noNPC = 0;
		if (npcPositions != null) {
			for (int i = 0; i < npcPositions.length; i++) {
				noNPC = npcPositions[i].size();
				if (npcPositions[i].size() > 0) {
					Vector2d closestNPCPos = npcPositions[i].get(0).position;
					double distToNPC = myPosition.dist(closestNPCPos);
					// to be a bit more *aggressive* on the gameplay, we will
					// make our heuristic move us *closer* to NPCs, regardless of
					// whether they are harmful or not
					stateVal -= distToNPC / 80;
				}
				if (npcPositions[i].size() > 1) {
					Vector2d farthestNPCPos = npcPositions[i].get(npcPositions[i].size()-1).position;
					double distToFar = myPosition.dist(farthestNPCPos);
					//stateVal -= distToFar / 50;
				}

			}
		}
		stateVal -= noNPC*300;

		return stateVal;
	}

	// at every 'state' the controller must make an action -- this method is thus called
	// this MUST return in 40 ms to correctly perform an action -- this is given by 'origTime'
	public Types.ACTIONS act(StateObservation origState, ElapsedCpuTimer origTime) {

		// generate arraylist of potential future states
		ArrayList<StateAndAncestor> population = new ArrayList<StateAndAncestor>();
		for (int i = 0; i < populationSize; i++) { population.add( new StateAndAncestor (origState.copy(),i)); }

		// for each of the generated states, make a randomized move, based on how many moves are available
		int numAvailMoves = origState.getAvailableActions().size();
		ArrayList<Types.ACTIONS> firstMove = new ArrayList<Types.ACTIONS>();
		for (int i = 0; i < populationSize; i++) {
			int actNo = rng.nextInt(numAvailMoves);
			//System.out.println("RNG: " + actNo);
			// populate an arraylist with one move per individual
			// it is good to remember the first move, so we can pick it later on
			firstMove.add( origState.getAvailableActions().get(actNo) ); 
			// actually apply the move to the copied state
			population.get(i).myState.advance( origState.getAvailableActions().get(actNo) );
		}

		int generationNo = 1;
		double remainingtime = origTime.remainingTimeMillis();
		int bestActor = 0; 	// the index of the individual in the population who is best

		ArrayList<StateTuple> stateScore = new ArrayList<StateTuple>();
		while ( generationNo < totalGenerations && remainingtime > 5.0 ) {
			stateScore.clear();
			// evaluate how good each of the individuals are
			for (int i = 0; i < populationSize; i++) {
				// state tuples have two things: an id (int) and a score (double)
				stateScore.add( new StateTuple( i, stateEval(population.get(i).myState) ) );
				remainingtime = origTime.remainingTimeMillis();
				if (remainingtime < 3.0) break;
			}

			remainingtime = origTime.remainingTimeMillis();
			if (remainingtime < 3.0) break;

			// pick the best 'mu' individuals and replace the remaining 'lambda' individuals
			ArrayList<StateTuple> toSortScores = new ArrayList<StateTuple>();

			for (StateTuple p : stateScore) { toSortScores.add( new StateTuple(p.stateno, p.statescore) ); }

			// this will sort them in ascending order of their score
			Collections.sort(toSortScores);

			int sortedsize = toSortScores.size();
			//for (int i = 0; i < sortedsize; i++)
				//System.out.println(toSortScores.get(i).stateno + " s: " + toSortScores.get(i).statescore);
			// the best actor is the ancestor of who is sorted the highest -- just keep track of index of who it is!
			bestActor = population.get(toSortScores.get(sortedsize-1).stateno).ancesNo;
			remainingtime = origTime.remainingTimeMillis();
			if (remainingtime < 3.0) break;
			// go through the lowest 'lambda' inviduals and replace them in the original population
			// note: we will only copy over the *best* state; this may limit diversity
			//System.out.println("-------");
			for (int i = 0; i < lamSize; i++) {
				
				// get indexes of the current individual (to replace) and index of the best individual
				int indexOfReplacedIndividual = toSortScores.get(i).stateno;
				int indexOfBestIndividual = toSortScores.get(populationSize-1).stateno;

				// copy state of the best individual, and descend it to our new individual (reproduction)
				// remember to retain memory of the best ancestor of this individual as well
				StateObservation stateOfBestIndividual = population.get(indexOfBestIndividual).myState.copy();
				int ancestorOfBestIndividual = population.get(indexOfBestIndividual).ancesNo;
				StateAndAncestor newIndividual = new StateAndAncestor(stateOfBestIndividual, ancestorOfBestIndividual);

				population.set( indexOfReplacedIndividual, newIndividual );
				remainingtime = origTime.remainingTimeMillis();
				if (remainingtime < 3.0) break;
			}

			remainingtime = origTime.remainingTimeMillis();
			if (remainingtime < 3.0) break;

			// now that everything is in the 'best' next state, we will generate another set of random actions to perform
			// we will perform a random action for each copied state (individual) to progress the tree search
			for (int i = 0; i < populationSize; i++) {
				int numMoves = population.get(i).myState.getAvailableActions().size();
				// this will happen if one of the individuals has died and no move remains
				if (numMoves > 0) {
					int moveSelect = rng.nextInt(numMoves);
					population.get(i).myState.advance( population.get(i).myState.getAvailableActions().get( moveSelect ) );
					//System.out.println("i: " + i + " new move: " + moveSelect);
				}
				remainingtime = origTime.remainingTimeMillis();
				if (remainingtime < 3.0) break;
			}
			generationNo++;
		}

		// Return the 'next action' of the best individual in the population
		Types.ACTIONS finalAction = firstMove.get(bestActor);

		//System.out.println("Best actor: " + bestActor);
		//System.out.println(origTime.remainingTimeMillis()); // if this is 0, then we are out of time

		//System.out.println("-------");
		//System.out.println("-------");

		return finalAction;
	}

}
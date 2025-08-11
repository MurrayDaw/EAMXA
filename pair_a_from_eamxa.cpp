/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Murray Daw - original implementation
                         Michael Chandross (SNL), Matthew Campbell (SNL) 
------------------------------------------------------------------------- */

// this version dated 30 Dec 2024 
// 28 Dec 2024 (MSD): loops made more efficient
// 30 Dec 2024 (MSD): short neighbor list  [time is 4.5 x EAM, compared to MEAM which is 7-8 x EAM]

#include "pair_a_from_eamxa.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairAFromEAMXA::PairAFromEAMXA(LAMMPS *lmp) : Pair(lmp), cut(nullptr)
{
  single_enable = 0;
  restartinfo = 0;

  cut_global = 0.;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairAFromEAMXA::settings(int narg, char **arg)
{

  if (narg != 1) error->all(FLERR, "Pair style AFromEAMXA must have exactly one argument (cutoff)");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  if(allocated){
    for (int i=1;i<=atom->ntypes;i++)
      for (int j=1;j<=atom->ntypes;j++) if(setflag[i][j])cut[i][j]=cut_global;
  }

}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairAFromEAMXA::coeff(int narg, char **arg)
{
  if (narg != 5 ) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated)allocate();

  int ilo, ihi, jlo, jhi;

  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  rin = utils::numeric(FLERR, arg[2], false, lmp);
  rout = utils::numeric(FLERR, arg[3], false, lmp);
  alpha = utils::numeric(FLERR, arg[4], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      cut[i][j] = cut_global;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
  
}


/* ---------------------------------------------------------------------- */

void PairAFromEAMXA::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;
  memory->create(setflag,np1,np1,"pair:setflag");
  for (int i=1; i<np1; i++)
    for (int j=1; j<np1; j++) setflag[i][j]=0;
  memory->create(cutsq,np1,np1,"pair:cutsq");
  memory->create(cut,np1,np1,"pair:cut");
}



/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairAFromEAMXA::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style afromeamxa requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this,NeighConst::REQ_FULL);

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairAFromEAMXA::init_one(int i, int j)
{

  if (setflag[i][j]==0) error->all(FLERR,"All pair coeffs are not set");

  return cut[i][j];
}
  

/* ---------------------------------------------------------------------- */

void PairAFromEAMXA::compute(int eflag, int vflag)
{

  int i, j, k, ii, jj, kk, inum, jnum;

  
  double ri[3], rj[3], rij[3], rhatij[3], rik[3], rhatik[3];
  double dijsq, dij, hij, hpij, dik, hik, hpik, gammajik, gjik, gpjik, u3jik;
  double mj, nj, mk, nk;
  double fi[3], fj[3], fk[3];
  double rji[3], rki[3];

  int *ilist, *jlist, *numneigh, **firstneigh;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;




  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  const double cutshortsq = rout*rout;

  const int maxshort = 40;
  int neighshort[maxshort];
  double rxshort[maxshort], ryshort[maxshort], rzshort[maxshort];
  double dshort[maxshort], rhatxshort[maxshort], rhatyshort[maxshort], rhatzshort[maxshort];
  double hshort[maxshort], hpshort[maxshort];
  int numshort;

  // loop over neighbors of my atoms 

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    ri[0] = x[i][0]; ri[1] = x[i][1]; ri[2] = x[i][2];

    jlist = firstneigh[i];  jnum = numneigh[i];

    // construct short neighbor list for atom i (to make double loops j,k shorter)
    // when combined with EAM, neighbor lists are constructed for that range
    // while range for 3-body part computed here is much shorter
    numshort = 0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      rj[0] = x[j][0]; rj[1] = x[j][1]; rj[2] = x[j][2];
      rij[0] = ri[0]-rj[0]; rij[1] = ri[1]-rj[1]; rij[2] = ri[2]-rj[2];
      dijsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];

      if ( dijsq < cutshortsq ) {

	neighshort[numshort] = j;
	rxshort[numshort] = rij[0];
	ryshort[numshort] = rij[1];
	rzshort[numshort] = rij[2];

	dij = sqrt(dijsq);
	dshort[numshort] = dij;
	rhatxshort[numshort] = rij[0]/dij;
	rhatyshort[numshort] = rij[1]/dij;
	rhatzshort[numshort] = rij[2]/dij;
	hshort[numshort] = h(dij);
	hpshort[numshort] = hp(dij);
	numshort++;

      }  // if j is in short range of i
    } // j loop constructing shortlist

    // now double loop over neighbors using short list 
    for ( jj = 0; jj < numshort-1; jj++) {       

      j = neighshort[jj];
      rij[0] = rxshort[jj];
      rij[1] = ryshort[jj];
      rij[2] = rzshort[jj];
      dij = dshort[jj];
      hij = hshort[jj];	hpij = hpshort[jj];
      rhatij[0] = rhatxshort[jj];
      rhatij[1] = rhatyshort[jj];
      rhatij[2] = rhatzshort[jj];

      for (kk = jj+1; kk < numshort; kk++) {

	k = neighshort[kk];
	rik[0] = rxshort[kk];
	rik[1] = ryshort[kk];
	rik[2] = rzshort[kk];
	dik = dshort[kk];
	hik = hshort[kk];	hpik = hpshort[kk];
	rhatik[0] = rhatxshort[kk];
	rhatik[1] = rhatyshort[kk];
	rhatik[2] = rhatzshort[kk];
	
	gammajik = rhatij[0]*rhatik[0] + rhatij[1]*rhatik[1] + rhatij[2]*rhatik[2];
	gjik = g(gammajik);
	gpjik = gp(gammajik);
	
	u3jik = -alpha*hij*hik*gjik;

	mj = -alpha*hik*(hpij*gjik-hij*gpjik*gammajik/dij);
	nj = -alpha*hik*hij*gpjik/dij;
	fj[0] = rhatij[0]*mj + rhatik[0]*nj;
	fj[1] = rhatij[1]*mj + rhatik[1]*nj;
	fj[2] = rhatij[2]*mj + rhatik[2]*nj;
	
	mk = -alpha*hij*(hpik*gjik-hik*gpjik*gammajik/dik);
	nk = -alpha*hij*hik*gpjik/dik;
	fk[0] = rhatik[0]*mk + rhatij[0]*nk;
	fk[1] = rhatik[1]*mk + rhatij[1]*nk;
	fk[2] = rhatik[2]*mk + rhatij[2]*nk;
	
	fi[0] = -fj[0]-fk[0]; // by translation symmetry
	fi[1] = -fj[1]-fk[1];
	fi[2] = -fj[2]-fk[2];
	
	f[i][0] += fi[0];
	f[i][1] += fi[1];
	f[i][2] += fi[2];
	
	f[j][0] += fj[0];
	f[j][1] += fj[1];
	f[j][2] += fj[2];
	
	f[k][0] += fk[0];
	f[k][1] += fk[1];
	f[k][2] += fk[2];
	
	/* preparing arguments for ev_tally3 */
	rji[0] = -rij[0];
	rji[1] = -rij[1];
	rji[2] = -rij[2];
	
	rki[0] = -rik[0];
	rki[1] = -rik[1];
	rki[2] = -rik[2];
	
	if (evflag) ev_tally3(i, j, k, u3jik, 0.0, fj, fk, rji, rki);
	
      } // k-loop (shortlist)
    } // j-loop (shortlist)
  } // i-loop

  if (vflag_fdotr) virial_fdotr_compute();

}


/* ---------------------------------------------------------------------- */

PairAFromEAMXA::~PairAFromEAMXA()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}



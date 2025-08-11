/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(afromeamxa,PairAFromEAMXA);
// clang-format on
#else
#ifndef LMP_PAIR_AFROMEAMXA_H
#define LMP_PAIR_AFROMEAMXA_H

#include "pair.h"
#include <cmath>

namespace LAMMPS_NS {

class PairAFromEAMXA : public Pair {
 public:
  
   PairAFromEAMXA(class LAMMPS *);
  
   ~PairAFromEAMXA() override;
  
   void compute(int, int) override;
   void settings(int, char **) override;
   void coeff(int, char **) override;
   void init_style() override;
   double init_one(int,int) override;

protected:
  double cut_global;
  double **cut;
  double rin, rout, alpha;

  double h(double r) {
    if (r < rin) {
      return 1.0;
    } else if (r > rout) {
      return 0.0;
    } else {
      return pow(r-rout,3)*(6*r*r+10*rin*rin-5*rin*rout+rout*rout+3*r*(rout-5*rin))/pow(rin-rout,5);
    }
  }

  double hp(double r) {
    if (r < rin || r> rout) {
      return 0.0;
    } else {
      return 30 * pow(r-rin,2) * pow(r-rout,2)/pow(rin-rout,5);
    }
  }


  double g(double gamma) {
    return pow(gamma,4);
  }

  double gp(double gamma) {
    return 4*pow(gamma,3);
  }

  double dot_product(double *vec1, double *vec2) {
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
  }
  
  virtual void allocate();
  
};
}    // namespace LAMMPS_NS
#endif
#endif

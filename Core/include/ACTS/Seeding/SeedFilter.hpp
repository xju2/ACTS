#pragma once

#include "ACTS/Seeding/InternalSeed.hpp"
#include "ACTS/Seeding/ISeedFilter.hpp"
#include "ACTS/Seeding/IQualityTool.hpp"

namespace Acts{
namespace Seeding{
  struct SeedFilterConfig{
    float deltaInvHelixRadius = 0.00003;
    float impactQualityFactor = 1.;
    float compatSeedQuality = 200.;
    float deltaRMin = 5.;
    int maxSeedsPerSpM = 10;
    int compatSeedLimit = 2;
  };

  class SeedFilter : public ISeedFilter{
    public: 
    SeedFilter(SeedFilterConfig cfg,
               std::shared_ptr<IQualityTool> qualityTool);

    SeedFilter() = delete;
    ~SeedFilter();

    virtual
    std::vector<std::pair<float, std::shared_ptr<InternalSeed> > >
    filterSeeds_2SpFixed(std::shared_ptr<SPForSeed> bottomSP,
                         std::shared_ptr<SPForSeed> middleSP,
                         std::vector<std::shared_ptr<SPForSeed>> topSpVec,
                         std::vector<float> invHelixRadiusVec,
                         std::vector<float> impactParametersVec,
                         float zOrigin) override;

    virtual
    std::vector<std::pair<float, std::shared_ptr<InternalSeed> > >
    filterSeeds_1SpFixed(std::vector<std::pair<float, std::shared_ptr<InternalSeed> > > seedsPerSpM) override;

    virtual
    std::vector<std::shared_ptr<Seed> >
    filterSeeds_byRegion(std::vector<std::pair<float, std::shared_ptr<InternalSeed> > > seedsPerRegion) override;

    private:
    const SeedFilterConfig m_cfg;
    const std::shared_ptr<IQualityTool> m_qualityTool;
  };
}
}


// quality comparator, returns true if first value is larger than second value
// using comQuality to sort will return higher quality -> lower quality.
class comQuality  {
public:
  
  bool operator ()
  (const std::pair<float,std::shared_ptr<Acts::Seeding::InternalSeed>>& i1,
   const std::pair<float,std::shared_ptr<Acts::Seeding::InternalSeed>>& i2)
  {
    return i1.first > i2.first;
  }
};

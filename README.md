# Ski Run Bearings / Trail Orientation / Aspect

WIP Ski Roses

## Development

```shell
# download latest OpenSkiMap data
poetry run ski_bearings download

# extract ski area metadata and metrics
poetry run ski_bearings analyze
```

## References

1. **Urban spatial order: street network orientation, configuration, and entropy**  
Geoff Boeing  
*Applied Network Science* (2019-08-23) <https://doi.org/gf8srn>  
DOI: [10.1007/s41109-019-0189-1](https://doi.org/10.1007/s41109-019-0189-1)

2. **OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks**  
Geoff Boeing  
*Computers, Environment and Urban Systems* (2017-09) <https://doi.org/gbvjxq>  
DOI: [10.1016/j.compenvurbsys.2017.05.004](https://doi.org/10.1016/j.compenvurbsys.2017.05.004)

3. https://github.com/russellporter/openskimap.org/issues/82

4. **Climate change exacerbates snow-water-energy challenges for European ski tourism**  
Hugues François, Raphaëlle Samacoïts, David Neil Bird, Judith Köberl, Franz Prettenthaler, Samuel Morin  
*Nature Climate Change* (2023-08-28) <https://doi.org/gsnhbv>  
DOI: [10.1038/s41558-023-01759-5](https://doi.org/10.1038/s41558-023-01759-5)

5. https://github.com/gboeing/osmnx/issues/1137

## Upstream issue tracking

- [openskimap.org/issues/82](https://github.com/russellporter/openskimap.org/issues/82): Add slope aspect information
- [openskimap.org/issues/135](https://github.com/russellporter/openskimap.org/issues/135): ski_areas.geojson location information is missing
- [openskimap.org/issues/137](https://github.com/russellporter/openskimap.org/issues/137): Restrict coordinate precision to prevent floating-point rounding errors
- [komoot/photon/issues/838](https://github.com/komoot/photon/issues/838) and [openskimap.org/issues/139](https://github.com/russellporter/openskimap.org/issues/139): Black Mountain of New Hampshire USA is missing location region metadata
- [osmnx/issues/1137](https://github.com/gboeing/osmnx/issues/1137) and [osmnx/pull/1139](https://github.com/gboeing/osmnx/pull/1139): Support directed bearing/orientation distributions and plots
- [osmnx/issues/1143](https://github.com/gboeing/osmnx/issues/1143) and [osmnx/pull/1147](https://github.com/gboeing/osmnx/pull/1147): _bearings_distribution: defer weighting to np.histogram
- [osmnx/pull/1149](https://github.com/gboeing/osmnx/pull/1149): _bearings_distribution: bin_centers terminology
- [JakobGM/patito/issues/103](https://github.com/JakobGM/patito/issues/103): Validation fails on an empty list
- [JakobGM/patito/issues/104](https://github.com/JakobGM/patito/issues/104): Optional list field with nested model fails to validate

## License

This project is built on data from [OpenSkiMap](https://openskimap.org/), which is based on [OpenStreetMap](https://www.openstreetmap.org/).
This data is released under the [Open Data Commons Open Database License](https://opendatacommons.org/licenses/odbl/).
Learn more at <https://www.openstreetmap.org/copyright>.

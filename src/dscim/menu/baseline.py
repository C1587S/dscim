import pandas as pd
import xarray as xr
from typing import Union
from dscim.menu.main_recipe import MainRecipe
import os

class Baseline(MainRecipe):
    """Adding up option"""

    NAME = "adding_up"
    __doc__ = MainRecipe.__doc__

    def ce_cc(self):
        pass

    def ce_no_cc(self):
        pass

    def global_damages_calculation(self) -> pd.DataFrame:
        """Call global damages

        Returns
        -------
        pd.DataFrame
            Global damages as DataFrame for compatibility with existing
            test suite and output pipelines.

        Notes
        -----
        Converts from the internal xarray DataArray representation
        (self.adding_up_damages) to DataFrame for output.
        """
        return self.adding_up_damages.to_dataframe("damages").reset_index()
    
    def damages_calculation(self, geography) -> xr.Dataset:
        """Aggregate damages to country level

        Returns
        --------
            pd.DataFrame
        """
        
        self.logger.info(f"Calculating damages")
        # For GWR discount types, use uncollapsed population for damage_function_points
        # to maintain backward compatibility with test expectations
        if "gwr" in self.discounting_type:
            pop_to_use = self.pop
        else:
            pop_to_use = self.collapsed_pop

        if self.individual_region:
            dams_collapse = self.calculated_damages * pop_to_use.sel(region = self.individual_region, drop=True)
        else:
            dams_collapse = self.calculated_damages * pop_to_use
        
        if geography == "ir":
            pass
        elif geography == "country":
            territories = []
            mapping_dict = {}
            for ii, row in self.countries_mapping.iterrows():
                mapping_dict[row["ISO"]] = row["MatchedISO"]
                if row["MatchedISO"] == "nan":
                    mapping_dict[row["ISO"]] = "nopop"
                    
            for region in dams_collapse.region.values:
                    territories.append(mapping_dict[region[:3]])
                    
            dams_collapse = (dams_collapse
                             .assign_coords({'region':territories})
                             .groupby('region')
                             .sum())
        elif geography == "globe":
            dams_collapse = dams_collapse.sum(dim="region").assign_coords({'region':'globe'}).expand_dims('region')

        return dams_collapse.to_dataset(name = 'damages')

    @property
    def calculated_damages(self):
        mean_cc = f"{self.ce_path}/adding_up_cc_eta{self.eta}.zarr"
        mean_no_cc = f"{self.ce_path}/adding_up_no_cc_eta{self.eta}.zarr"

        if os.path.exists(mean_cc) and os.path.exists(mean_no_cc):
            self.logger.info(
                f"Adding up aggregated damages found at {mean_cc}, {mean_no_cc}. These are being loaded..."
            )
            damages = (
                (xr.open_zarr(mean_no_cc).no_cc - xr.open_zarr(mean_cc).cc)
            )
        else:
            raise NotImplementedError(
                f"Adding up reduced damages not found: {mean_no_cc}, {mean_cc}. Please reduce damages for for `adding_up`."
            )
        if self.individual_region:
            damages = damages.sel(region=self.individual_region,drop=True)
        return self.cut(damages)

    def ce_cc_calculation(self):
        pass

    def ce_test(self):
        pass

    def ce_no_cc_calculation(self):
        pass

    def global_consumption_calculation(self, disc_type):
        """Calculate global consumption"""

        if self.geography == 'ir':
            gdp = self.gdp
        elif self.geography == 'country':
            #group gdp by some grouping and collapse
            gdp = self.gdp.sum(dim=["region"])
        else:
            gdp = self.gdp.sum(dim=["region"])

        if (disc_type == "constant") or ("ramsey" in disc_type):
            global_cons_no_cc = gdp

        elif disc_type == "constant_model_collapsed":
            global_cons_no_cc = gdp.mean(dim=["model"])

        elif "gwr" in disc_type:
            global_cons_no_cc = gdp.mean(dim=["model", "ssp"])

        global_cons_no_cc.name = f"global_cons_{disc_type}"

        return global_cons_no_cc
    
#         def global_consumption_calculation(self, disc_type):
#         """Calculate global consumption

#         Returns
#         -------
#             xr.DataArray
#         """

#         if self.geography == 'ir':
#             gdp = self.gdp
#         elif self.geography == 'country':
#             #group gdp by some grouping and collapse
#             gdp = self.gdp.sum(dim=["region"])
#         else:
#             gdp = self.gdp.sum(dim=["region"])

#         if (disc_type == "constant") or ("ramsey" in disc_type):
#             global_cons_no_cc = gdp

#         elif disc_type == "constant_model_collapsed":
#             global_cons_no_cc = gdp.mean(dim=["model"])

#         elif "gwr" in disc_type:
#             ce_cons = self.ce(self.gdppc, dims=["ssp", "model"])
#             global_cons_no_cc = (ce_cons * self.collapsed_pop).sum(dim=["region"])

#         # Convert to array in case xarray becames temperamental. This is a hack
#         # that need to be changed
#         if isinstance(global_cons_no_cc, xr.Dataset):
#             global_cons_no_cc = global_cons_no_cc.to_array()

#         global_cons_no_cc.name = f"global_cons_{disc_type}"

#         return global_cons_no_cc

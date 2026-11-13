// SPDX-License-Identifier: MIT
// Clean-room WebGL1 Live 2.5D Dexel Stock Carving, Feeds & Speeds Chipload, & 3-Way Side-by-Side Simulator.
// TODO(security): Scoped as a read-only local 127.0.0.1 single-user viewer with zero cookies,
// zero storage APIs, zero inline styles, and strict createElement/createElementNS/textContent DOM construction.

(function () {
  "use strict";

  const MAX_STL_TRIANGLES = 2000000;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const DEXEL_PITCH_MM = 0.35;
  const KEYFRAME_STRIDE = 16;

  const INCH_MM = 25.4;
  const HP_TO_KW = 0.745699872;
  const LBF_IN_TO_NM = 0.112984829;
  const BLENDED_EPS_LEN_MM = 1e-4;
  const BLENDED_EPS_VEL_MM_S = 1e-4;

  const STOCK_PALETTES = {
    green: { skin: [0.22, 0.56, 0.26], alloy: [0.73, 0.79, 0.86], wall: [0.64, 0.71, 0.79] },
    silver: { skin: [0.56, 0.62, 0.69], alloy: [0.80, 0.84, 0.90], wall: [0.68, 0.73, 0.80] },
    blue: { skin: [0.20, 0.44, 0.68], alloy: [0.75, 0.80, 0.87], wall: [0.64, 0.71, 0.79] },
    amber: { skin: [0.68, 0.49, 0.18], alloy: [0.76, 0.81, 0.87], wall: [0.66, 0.72, 0.79] },
  };

  // Built-in clean-room Feeds & Speeds catalog (mirrored 1:1 with cam/feeds_and_speeds.py)
  const DEFAULT_FS_CATALOG = {
    inch_mm: INCH_MM,
    endmill_diameter_ipt_table_in: [
      [0.03125, 0.00015],
      [0.06250, 0.00032],
      [0.09375, 0.00048],
      [0.12500, 0.00065],
      [0.18750, 0.00097],
      [0.25000, 0.00130],
      [0.31250, 0.00168],
      [0.37500, 0.00206],
      [0.50000, 0.00285],
      [0.62500, 0.00340],
      [0.75000, 0.00390],
      [1.00000, 0.00460],
    ],
    stock_materials: [
      { id: "al_6061_t6", label: "Aluminum 6061-T6", category: "Non-Ferrous Alloy", hardness_hb: 95.0, kp_hp_min_in3: 0.25, sfm_carbide: 850.0, ipt_hss_in: 0.00320, ipt_carbide_in: 0.00360, material_ipt_reduction: 0.15, max_ipt_in: 0.00650, ideal_woc_frac: 0.35, ideal_doc_frac: 0.65 },
      { id: "al_7075_t6", label: "Aluminum 7075-T6", category: "Non-Ferrous Alloy", hardness_hb: 150.0, kp_hp_min_in3: 0.30, sfm_carbide: 700.0, ipt_hss_in: 0.00290, ipt_carbide_in: 0.00330, material_ipt_reduction: 0.20, max_ipt_in: 0.00600, ideal_woc_frac: 0.30, ideal_doc_frac: 0.65 },
      { id: "brass_c360", label: "Brass C360 (Free-Machining)", category: "Copper Alloy", hardness_hb: 100.0, kp_hp_min_in3: 0.32, sfm_carbide: 600.0, ipt_hss_in: 0.00300, ipt_carbide_in: 0.00340, material_ipt_reduction: 0.18, max_ipt_in: 0.00620, ideal_woc_frac: 0.35, ideal_doc_frac: 0.65 },
      { id: "delrin_pom", label: "Acetal / Delrin (POM)", category: "Engineering Polymer", hardness_hb: 40.0, kp_hp_min_in3: 0.08, sfm_carbide: 1000.0, ipt_hss_in: 0.00390, ipt_carbide_in: 0.00440, material_ipt_reduction: 0.10, max_ipt_in: 0.00800, ideal_woc_frac: 0.45, ideal_doc_frac: 0.80 },
      { id: "steel_1018", label: "Mild Steel 1018 / A36", category: "Carbon Steel", hardness_hb: 135.0, kp_hp_min_in3: 0.85, sfm_carbide: 380.0, ipt_hss_in: 0.00240, ipt_carbide_in: 0.00276, material_ipt_reduction: 0.35, max_ipt_in: 0.00500, ideal_woc_frac: 0.20, ideal_doc_frac: 0.60 },
      { id: "steel_4140", label: "Alloy Steel 4140 (28 HRC)", category: "Alloy Steel", hardness_hb: 280.0, kp_hp_min_in3: 1.15, sfm_carbide: 250.0, ipt_hss_in: 0.00170, ipt_carbide_in: 0.00210, material_ipt_reduction: 0.45, max_ipt_in: 0.00400, ideal_woc_frac: 0.12, ideal_doc_frac: 0.55 },
      { id: "ss_304", label: "Stainless Steel 304", category: "Austenitic Stainless", hardness_hb: 190.0, kp_hp_min_in3: 1.25, sfm_carbide: 220.0, ipt_hss_in: 0.00180, ipt_carbide_in: 0.00220, material_ipt_reduction: 0.50, max_ipt_in: 0.00400, ideal_woc_frac: 0.12, ideal_doc_frac: 0.55 },
      { id: "ti_6al4v", label: "Titanium Ti-6Al-4V", category: "Titanium Alloy", hardness_hb: 330.0, kp_hp_min_in3: 1.35, sfm_carbide: 150.0, ipt_hss_in: 0.00140, ipt_carbide_in: 0.00180, material_ipt_reduction: 0.55, max_ipt_in: 0.00320, ideal_woc_frac: 0.10, ideal_doc_frac: 0.50 },
    ],
    tool_materials: [
      { id: "carbide_coated", label: "Coated Carbide (TiAlN / ZrN)", is_carbide: true, sfm_mult: 1.18, ipt_mult: 1.05, coat_ipt_mult: 1.05, youngs_gpa: 600.0, yield_ksi: 85.0, dev_max_ratio: 0.0012 },
      { id: "carbide_uncoated", label: "Solid Carbide (Uncoated)", is_carbide: true, sfm_mult: 1.00, ipt_mult: 1.00, coat_ipt_mult: 1.00, youngs_gpa: 600.0, yield_ksi: 85.0, dev_max_ratio: 0.0012 },
      { id: "cobalt_m42", label: "Cobalt HSS (M42 8% Co)", is_carbide: false, sfm_mult: 0.42, ipt_mult: 0.82, coat_ipt_mult: 1.00, youngs_gpa: 215.0, yield_ksi: 60.0, dev_max_ratio: 0.0018 },
      { id: "hss_m2", label: "High-Speed Steel (HSS M2)", is_carbide: false, sfm_mult: 0.30, ipt_mult: 0.72, coat_ipt_mult: 1.00, youngs_gpa: 205.0, yield_ksi: 52.0, dev_max_ratio: 0.0020 },
    ],
    default_machine: {
      min_rpm: 100.0,
      max_rpm: 8000.0,
      max_feed_mm_s: 50.0,
      rapid_xy_mm_s: 120.0,
      rapid_z_mm_s: 80.0,
      spindle_power_kw: 5.6,
      max_machine_torque_nm: 23.0,
      max_cutting_force_n: 1200.0,
    },
  };

  // --- Clean-Room Feeds, Speeds, Chip Thinning & Physical Constraint Solver ---
  function lerpClamped(x1, x2, y1, y2, x) {
    if (Math.abs(x2 - x1) <= 1e-15) return Number(y1);
    let t = (x - x1) / (x2 - x1);
    if (t <= 0.0) t = 0.0;
    else if (t >= 1.0) t = 1.0;
    return Number(y1 + t * (y2 - y1));
  }

  function reduceIptFactor(toolIptFactor, materialReduction) {
    return 1.0 + (1.0 - materialReduction) * (toolIptFactor - 1.0);
  }

  function lookupEndmillReferenceIptIn(diameterIn, tableOverride) {
    const table = tableOverride || DEFAULT_FS_CATALOG.endmill_diameter_ipt_table_in;
    const d = Number(diameterIn);
    if (d <= table[0][0]) return Number(table[0][1]);
    if (d >= table[table.length - 1][0]) return Number(table[table.length - 1][1]);
    for (let i = 0; i < table.length - 1; i++) {
      const d0 = table[i][0], q0 = table[i][1];
      const d1 = table[i + 1][0], q1 = table[i + 1][1];
      if (d0 <= d && d <= d1) return lerpClamped(d0, d1, q0, q1, d);
    }
    return Number(table[table.length - 1][1]);
  }

  function radialChipThinningMultiplier(wocMm, effDiameterMm) {
    if (wocMm <= 0.0 || effDiameterMm <= 1e-9 || 2.0 * wocMm >= effDiameterMm) return 1.0;
    const u = 1.0 - (2.0 * wocMm) / effDiameterMm;
    const denom = Math.sqrt(Math.max(1.0 - u * u, 1e-9));
    return Math.min(3.0, 1.0 / denom);
  }

  function radialChipThinningPsi(uAe) {
    if (uAe <= 0.0) return 0.0;
    if (uAe >= 0.5) return 1.0;
    return Math.sqrt(Math.max(0.0, 4.0 * uAe * (1.0 - uAe)));
  }

  function axialChipThinningMultiplier(docMm, cornerRadiusMm) {
    if (cornerRadiusMm <= 1e-6 || docMm <= 0.0 || docMm >= cornerRadiusMm) return 1.0;
    const u = 1.0 - docMm / cornerRadiusMm;
    const denom = Math.sqrt(Math.max(1.0 - u * u, 1e-9));
    return Math.min(10.0, 1.0 / denom);
  }

  function computeNominalBaseIptIn(stock, toolMat, cutter, wocMm, iptTable) {
    const dIn = Math.max(cutter.diameter_mm / INCH_MM, 1e-5);
    const wocIn = Math.max(0.0, wocMm / INCH_MM);
    const qD = lookupEndmillReferenceIptIn(dIn, iptTable);
    const qRef = lookupEndmillReferenceIptIn(0.50, iptTable);
    const diameterRatio = qD / Math.max(qRef, 1e-12);

    const helixRad = (cutter.helix_deg * Math.PI) / 180.0;
    const helixFactor = 1.0 + 0.5 * (Math.sin(helixRad) - 0.5);

    const idealSideWocIn = stock.ideal_woc_frac * dIn;
    const slotFactor =
      wocIn <= 0.0
        ? 1.0
        : lerpClamped(idealSideWocIn, dIn, 1.0, cutter.slot_ipt_factor, wocIn);

    let materialChip;
    if (toolMat.is_carbide && stock.ipt_carbide_in > 0.0) {
      materialChip = stock.ipt_carbide_in * toolMat.coat_ipt_mult;
    } else {
      materialChip = stock.ipt_hss_in * reduceIptFactor(toolMat.ipt_mult, stock.material_ipt_reduction);
    }

    let fzBaseIn = diameterRatio * slotFactor * materialChip * helixFactor * cutter.tool_type_ipt;
    if (stock.max_ipt_in > 0.0) {
      fzBaseIn = Math.min(fzBaseIn, stock.max_ipt_in);
    }
    return { fzBaseIn, slotFactor };
  }

  function computeFeedsAndSpeeds(params, catalogOverride) {
    const cat = catalogOverride || DEFAULT_FS_CATALOG;
    const stockList = cat.stock_materials || DEFAULT_FS_CATALOG.stock_materials;
    const toolList = cat.tool_materials || DEFAULT_FS_CATALOG.tool_materials;
    const stock = stockList.find((m) => m.id === params.stock_material_id) || stockList[0];
    const toolMat = toolList.find((t) => t.id === params.tool_material_id) || toolList[0];

    const cutter = Object.assign(
      {
        diameter_mm: 5.0,
        flutes: 3,
        flute_length_mm: 12.0,
        stickout_mm: 14.0,
        shank_diameter_mm: 5.0,
        corner_radius_mm: 0.0,
        helix_deg: 35.0,
        slot_ipt_factor: 0.80,
        tool_type_ipt: 1.0,
        max_rpm: 24000.0,
        max_deflection_pct: 100.0,
        max_torque_pct: 100.0,
      },
      params.cutter || {}
    );
    const machine = Object.assign({}, DEFAULT_FS_CATALOG.default_machine, cat.default_machine || {}, params.machine || {});

    const aggr = Math.max(0.25, Math.min(2.50, Number(params.aggressiveness ?? 1.0)));
    const dMm = Math.max(0.5, Number(cutter.diameter_mm));
    const dIn = dMm / INCH_MM;
    const docCleanMm = Math.max(0.0, Math.min(Number(params.doc_mm ?? 3.0), cutter.flute_length_mm * 1.5));
    const wocCleanMm = Math.max(0.0, Math.min(Number(params.woc_mm ?? 1.75), dMm));
    const docIn = docCleanMm / INCH_MM;
    const wocIn = wocCleanMm / INCH_MM;

    const chipThinningEnabled = params.chip_thinning_enabled !== false;
    const engagementLoadEnabled = params.engagement_load_enabled !== false;
    const hsmSpeedEnabled = params.hsm_speed_enabled !== false;

    const baseObj = computeNominalBaseIptIn(stock, toolMat, cutter, wocCleanMm, cat.endmill_diameter_ipt_table_in);
    const fzBaseIn = baseObj.fzBaseIn;
    const slotFactor = baseObj.slotFactor;

    const rThin = radialChipThinningMultiplier(wocCleanMm, dMm);
    const aThin = axialChipThinningMultiplier(docCleanMm, cutter.corner_radius_mm);
    const cThinAll = rThin * aThin;
    const cThin = chipThinningEnabled ? cThinAll : 1.0;

    const idealWocIn = stock.ideal_woc_frac * dIn;
    const idealDocIn = stock.ideal_doc_frac * dIn;
    let cLoad = 1.0;
    if (engagementLoadEnabled && docIn * wocIn > 1e-8) {
      const aTarget = idealDocIn * idealWocIn;
      const cLoadRaw = aTarget / (docIn * wocIn);
      let kMax =
        lerpClamped(0.0, 3.0, 1.375, 1.2, stock.kp_hp_min_in3) *
        lerpClamped(1.0, 3.0, 1.5, 1.0, cThinAll);
      kMax = Math.max(1.0, Math.min(6.0, kMax));
      cLoad = Math.max(0.5, Math.min(kMax, cLoadRaw));
    }

    const rpmCap = Math.min(machine.max_rpm, cutter.max_rpm);
    let rpm;
    let sfm;
    let speedLimitTag = "";
    if (params.fixed_spindle_rpm && Number(params.fixed_spindle_rpm) > 0.0) {
      rpm = Math.max(machine.min_rpm, Math.min(rpmCap, Number(params.fixed_spindle_rpm)));
      sfm = (rpm * Math.PI * dIn) / 12.0;
    } else {
      const sfmSlot = wocIn > 0.0 ? lerpClamped(idealWocIn, dIn, 1.0, 0.85, wocIn) : 1.0;
      const hsmBoost =
        hsmSpeedEnabled && wocCleanMm > 0.0 && 2.0 * wocCleanMm < dMm
          ? Math.min(rThin * lerpClamped(1.0, 15.0, 1.0, 1.5, aThin), 1.5)
          : 1.0;
      const speedAggr = 0.75 + 0.25 * aggr;
      const sfmTarget = stock.sfm_carbide * toolMat.sfm_mult * sfmSlot * hsmBoost * speedAggr;
      const rpmRaw = (12.0 * sfmTarget) / (Math.PI * Math.max(dIn, 1e-6));
      if (rpmRaw > rpmCap) {
        rpm = Number(rpmCap);
        speedLimitTag = "Spindle Max RPM";
      } else if (rpmRaw < machine.min_rpm) {
        rpm = Number(machine.min_rpm);
        speedLimitTag = "Spindle Min RPM";
      } else {
        rpm = Math.round(rpmRaw);
      }
      sfm = (rpm * Math.PI * dIn) / 12.0;
    }

    let fzProgIn = fzBaseIn * cThin * cLoad * aggr;
    if (stock.max_ipt_in > 0.0) {
      const maxActualIn = stock.max_ipt_in * Math.max(1.0, 0.65 + 0.35 * aggr);
      if (fzProgIn / Math.max(cThinAll, 1.0) > maxActualIn) {
        fzProgIn = maxActualIn * cThin;
      }
    }

    const feedInMin = rpm * cutter.flutes * fzProgIn;
    let feedMmMin = feedInMin * INCH_MM;
    let feedMmS = feedMmMin / 60.0;

    const mrrIn3Min = docIn * wocIn * feedInMin;
    let powerKw = mrrIn3Min * stock.kp_hp_min_in3 * HP_TO_KW;
    const omegaRadS = Math.max((2.0 * Math.PI * rpm) / 60.0, 1e-6);
    let torqueNm = (powerKw * 1000.0) / omegaRadS;
    const rEffM = Math.max(0.5 * dMm * 1e-3, 1e-5);
    const tangentialN = torqueNm / rEffM;
    let cuttingForceN = 1.15 * tangentialN;

    const shankMm = Math.max(0.5, cutter.shank_diameter_mm);
    const dCoreIn = 0.58 * (Math.min(dMm, shankMm) / INCH_MM);
    const shaftYieldLbfIn = (Math.PI / 16.0) * (1000.0 * toolMat.yield_ksi) * Math.pow(dCoreIn, 3);
    const toolMaxTorqueNm = shaftYieldLbfIn * LBF_IN_TO_NM * 0.35 * (cutter.max_torque_pct / 100.0);

    const dStiffMm = 0.78 * Math.min(dMm, shankMm);
    const inertiaMm4 = (Math.PI / 64.0) * Math.pow(dStiffMm, 4);
    const youngsNMm2 = toolMat.youngs_gpa * 1000.0;
    let deflectionMm =
      (cuttingForceN * Math.pow(cutter.stickout_mm, 3)) /
      Math.max(3.0 * youngsNMm2 * inertiaMm4, 1e-9);
    const maxDeflectionMm = Math.max(
      0.005,
      toolMat.dev_max_ratio * cutter.stickout_mm * (cutter.max_deflection_pct / 100.0)
    );

    let uScale = 1.0;
    let feedLimitTag = "";
    const candidates = [];
    if (feedMmS > 1e-9) candidates.push([(machine.max_feed_mm_s * 0.999) / feedMmS, "Machine Max Feed"]);
    if (powerKw > 1e-9) candidates.push([machine.spindle_power_kw / powerKw, "Spindle Power"]);
    if (torqueNm > 1e-9) {
      candidates.push([(machine.max_machine_torque_nm * 0.99) / torqueNm, "Spindle Torque"]);
      candidates.push([(toolMaxTorqueNm * 0.99) / torqueNm, "Tool Torque"]);
    }
    if (cuttingForceN > 1e-9) candidates.push([(machine.max_cutting_force_n * 0.99) / cuttingForceN, "Cutting Force"]);
    if (deflectionMm > 1e-9) candidates.push([(maxDeflectionMm * 0.99) / deflectionMm, "Tool Deflection"]);

    for (let i = 0; i < candidates.length; i++) {
      if (candidates[i][0] < uScale) {
        uScale = candidates[i][0];
        feedLimitTag = candidates[i][1];
      }
    }

    uScale = Math.max(0.02, Math.min(1.0, uScale));
    fzProgIn *= uScale;
    feedMmS *= uScale;
    feedMmMin *= uScale;
    powerKw *= uScale;
    torqueNm *= uScale;
    cuttingForceN *= uScale;
    deflectionMm *= uScale;
    const mrrMm3S = docCleanMm * wocCleanMm * feedMmS;
    const chipThicknessIn = fzProgIn / Math.max(cThinAll, 1.0);
    const activeLimit = feedLimitTag || speedLimitTag || "Optimal Chipload";

    return {
      stock_material_id: stock.id,
      stock_label: stock.label,
      tool_material_id: toolMat.id,
      tool_label: toolMat.label,
      aggressiveness: Number(aggr.toFixed(4)),
      diameter_mm: Number(dMm.toFixed(4)),
      flutes: Math.round(cutter.flutes),
      stickout_mm: Number(cutter.stickout_mm),
      doc_mm: Number(docCleanMm.toFixed(4)),
      woc_mm: Number(wocCleanMm.toFixed(4)),
      spindle_rpm: Number(rpm.toFixed(1)),
      surface_speed_sfm: Number(sfm.toFixed(2)),
      surface_speed_m_min: Number((sfm * 0.3048).toFixed(2)),
      fz_base_mm: Number((fzBaseIn * INCH_MM).toFixed(6)),
      fz_programmed_mm: Number((fzProgIn * INCH_MM).toFixed(6)),
      chip_thickness_mm: Number((chipThicknessIn * INCH_MM).toFixed(6)),
      radial_thinning_factor: Number(rThin.toFixed(5)),
      axial_thinning_factor: Number(aThin.toFixed(5)),
      slot_factor: Number(slotFactor.toFixed(5)),
      engagement_load_factor: Number(cLoad.toFixed(5)),
      feed_mm_s: Number(feedMmS.toFixed(4)),
      feed_mm_min: Number(feedMmMin.toFixed(2)),
      mrr_mm3_s: Number(mrrMm3S.toFixed(3)),
      spindle_power_kw: Number(powerKw.toFixed(4)),
      power_utilization_pct: Number(((100.0 * powerKw) / Math.max(machine.spindle_power_kw, 1e-6)).toFixed(2)),
      torque_nm: Number(torqueNm.toFixed(4)),
      cutting_force_n: Number(cuttingForceN.toFixed(2)),
      deflection_mm: Number(deflectionMm.toFixed(5)),
      max_deflection_mm: Number(maxDeflectionMm.toFixed(5)),
      deflection_utilization_pct: Number(((100.0 * deflectionMm) / Math.max(maxDeflectionMm, 1e-6)).toFixed(2)),
      constraint_feed_scale: Number(uScale.toFixed(5)),
      active_limit: activeLimit,
    };
  }

  // Exact mirror of benchmarks/contracts.py::blended_segment_durations
  function blendedSegmentDurations(segments, speedsMmS, maxAccelMmS2) {
    const m = segments.length;
    const out = new Float64Array(m);
    if (m === 0) return out;
    const inv2A = 0.5 / Math.max(Number(maxAccelMmS2 || 800.0), 1.0);
    let prevVx = 0.0, prevVy = 0.0, prevVz = 0.0;
    let lastVx = 0.0, lastVy = 0.0, lastVz = 0.0;

    for (let k = 0; k < m; k++) {
      const s = segments[k];
      const dx = s.end[0] - s.start[0];
      const dy = s.end[1] - s.start[1];
      const dz = s.end[2] - s.start[2];
      const spd = Math.max(0.2, Number(speedsMmS[k]));
      const normEps = Math.sqrt(dx * dx + dy * dy + dz * dz + BLENDED_EPS_LEN_MM * BLENDED_EPS_LEN_MM);
      const effLen = normEps - BLENDED_EPS_LEN_MM;
      const cruise = effLen / spd;
      const vx = (spd / normEps) * dx;
      const vy = (spd / normEps) * dy;
      const vz = (spd / normEps) * dz;
      const dvx = vx - prevVx;
      const dvy = vy - prevVy;
      const dvz = vz - prevVz;
      const jNorm = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz + BLENDED_EPS_VEL_MM_S * BLENDED_EPS_VEL_MM_S);
      out[k] = cruise + (jNorm - BLENDED_EPS_VEL_MM_S) * inv2A;
      prevVx = vx; prevVy = vy; prevVz = vz;
      if (k === m - 1) {
        lastVx = vx; lastVy = vy; lastVz = vz;
      }
    }
    const endNorm = Math.sqrt(lastVx * lastVx + lastVy * lastVy + lastVz * lastVz + BLENDED_EPS_VEL_MM_S * BLENDED_EPS_VEL_MM_S);
    out[m - 1] += (endNorm - BLENDED_EPS_VEL_MM_S) * inv2A;
    return out;
  }

  function buildCutterSpecFromToolRecord(t) {
    const src = t || {};
    const diam = Math.max(0.5, Number(src.diameter_mm || (src.cutter_radius_mm ? 2.0 * src.cutter_radius_mm : 5.0)));
    const fluteLen = Math.max(1.0, Number(src.flute_length_mm || 12.0));
    const stickout = Math.max(fluteLen + 1.5, Number(src.stickout_mm || src.holder_bottom_mm || 14.0));
    const shankDiam = Math.max(0.5, Number(src.shank_diameter_mm || (src.shank_radius_mm ? 2.0 * src.shank_radius_mm : diam)));
    const etype = String(src.entity_type || "flat end mill").toLowerCase();
    const isFace = etype === "face mill";
    return {
      diameter_mm: diam,
      flutes: Math.max(1, Number(src.flutes || 3)),
      flute_length_mm: fluteLen,
      stickout_mm: stickout,
      shank_diameter_mm: shankDiam,
      corner_radius_mm: Math.max(0.0, Number(src.corner_radius_mm || 0.0)),
      helix_deg: Number(src.helix_deg !== undefined ? src.helix_deg : 35.0),
      slot_ipt_factor: isFace ? 0.90 : 0.80,
      tool_type_ipt: isFace ? 1.15 : 1.0,
      max_rpm: Math.max(1000.0, Number(src.max_rpm || 24000.0)),
      max_deflection_pct: 100.0,
      max_torque_pct: 100.0,
      entity_type: etype,
    };
  }

  function buildCutterSpecFromCase(caseData) {
    const tbl = (caseData && caseData.tools_table) || null;
    const t1 = tbl && (tbl["1"] || tbl[Object.keys(tbl)[0]]);
    return buildCutterSpecFromToolRecord(t1 || (caseData && caseData.tool) || {});
  }

  function isAxialHolemakingToolType(etype) {
    const e = String(etype || "").toLowerCase();
    return (
      e === "drill" ||
      e === "spot drill" ||
      e === "center drill" ||
      e === "reamer" ||
      e === "tap right hand" ||
      e === "probe"
    );
  }

  // Re-evaluate a variant's segment speeds, durations, and chipload under current Fair / Optimal Chipload policy
  function evaluateVariantUnderPolicy(variant, caseData, settings, baseRes, catalog) {
    const origPlan = variant.origPlan || variant.plan;
    if (!variant.origPlan) {
      variant.origPlan = JSON.parse(JSON.stringify(variant.plan));
    }
    const srcSegs = variant.origPlan.segments || [];
    const primaryCutter = buildCutterSpecFromCase(caseData);
    const toolsTable = (caseData && caseData.tools_table) || null;
    const rapidSpeed = Number(caseData.rapid_mm_s || 120.0);
    const maxAccel = Number(caseData.max_accel_mm_s2 || 1500.0);
    const nomDoc = Number(caseData.nominal_doc_mm || 3.0);
    const nomWoc = Number(caseData.nominal_woc_mm || 1.75);

    // Precompute per-tool baseline feeds/speeds so multi-tool cases respect each tool's physical diameter & flutes
    const toolBaselineCache = new Map();
    function getToolBaseline(tNum) {
      const key = Number(tNum || 1);
      if (toolBaselineCache.has(key)) return toolBaselineCache.get(key);
      const tRec = toolsTable && (toolsTable[String(key)] || toolsTable[key]);
      if (!tRec) {
        const fallback = {
          cutter: primaryCutter,
          res: baseRes,
          isHole: false,
          rec: (caseData && caseData.tool) || {},
        };
        toolBaselineCache.set(key, fallback);
        return fallback;
      }
      const cSpec = buildCutterSpecFromToolRecord(tRec);
      const isHole = isAxialHolemakingToolType(cSpec.entity_type);
      const tDoc = Math.min(nomDoc, 0.7 * cSpec.diameter_mm);
      const tWoc = Math.min(nomWoc, 0.38 * cSpec.diameter_mm);
      const tRes = computeFeedsAndSpeeds(
        {
          stock_material_id: settings.stockMaterialId,
          tool_material_id: settings.toolMaterialId,
          cutter: cSpec,
          doc_mm: Math.max(0.25, tDoc),
          woc_mm: Math.max(0.10, tWoc),
          aggressiveness: settings.aggressiveness,
        },
        catalog
      );
      const entry = { cutter: cSpec, res: tRes, isHole, rec: tRec };
      toolBaselineCache.set(key, entry);
      return entry;
    }

    const baseFeedPrimary = Math.max(1.0, Number(baseRes.feed_mm_s));
    const spindleRpmPrimary = Math.max(100.0, Number(baseRes.spindle_rpm));
    const hOptPrimary = Math.max(0.004, Number(baseRes.chip_thickness_mm));

    const isNaive = variant.id === "nominal_cam";
    const isRl = variant.id === "shac_chipload" || variant.id === "shac_policy";
    const origMeanCut = Math.max(1.0, Number(origPlan.mean_cut_feed_mm_s || 10.0));

    const speeds = new Float64Array(srcSegs.length);
    for (let i = 0; i < srcSegs.length; i++) {
      const s = srcSegs[i];
      if (s.kind === "rapid") {
        speeds[i] = rapidSpeed;
        continue;
      }
      const tb = getToolBaseline(s.tool_number || 1);
      if (tb.isHole) {
        // Axial drills / spot drills / taps scale cleanly around their recorded plunge feedrate
        const origPlunge = Math.max(0.5, Number(s.feed_mm_s || 4.0));
        speeds[i] = Math.max(0.5, Math.min(25.0, origPlunge * settings.aggressiveness));
        continue;
      }
      const baseFeedS = Math.max(1.0, Number(tb.res.feed_mm_s));
      const spindleRpm = Math.max(100.0, Number(tb.res.spindle_rpm));

      if (settings.speedPolicy === "matched" || isNaive) {
        // Fair Same-Speed baseline OR Naive constant-feed CAM in Optimization Mode
        speeds[i] = baseFeedS;
        continue;
      }
      if (!s.engaged) {
        // Unengaged air G1 station in Optimization Mode: sprint within machine/RL envelope
        if (isRl) {
          const rlMult = Math.max(0.55, Math.min(2.1, Number(s.feed_mm_s || origMeanCut) / origMeanCut));
          speeds[i] = Math.min(48.0, baseFeedS * Math.max(1.15, rlMult));
        } else {
          speeds[i] = Math.min(48.0, Math.max(baseFeedS * 2.6, 30.0));
        }
        continue;
      }

      const apLocal = Math.max(0.25, Number(s.ap_mm || nomDoc));
      const aeLocal = Math.max(0.08, Number(s.ae_mm || nomWoc));
      const localRes = computeFeedsAndSpeeds(
        {
          stock_material_id: settings.stockMaterialId,
          tool_material_id: settings.toolMaterialId,
          cutter: tb.cutter,
          doc_mm: apLocal,
          woc_mm: aeLocal,
          aggressiveness: settings.aggressiveness,
          fixed_spindle_rpm: spindleRpm,
        },
        catalog
      );

      if (isRl) {
        // RL neural policy modulates around baseline feed, bounded by local physics ceiling
        const rlRatio = Math.max(0.45, Math.min(1.95, Number(s.feed_mm_s || origMeanCut) / origMeanCut));
        const rlFeed = baseFeedS * rlRatio;
        speeds[i] = Math.max(1.5, Math.min(rlFeed, localRes.feed_mm_s * 1.18, 48.0));
      } else {
        // Direct DiffCAM: exact station-level optimal chipload feed from local (a_p, a_e)
        speeds[i] = Math.max(1.5, Number(localRes.feed_mm_s));
      }
    }

    const durs = blendedSegmentDurations(srcSegs, speeds, maxAccel);
    const newSegs = new Array(srcSegs.length);
    let tCursor = 0.0;
    let totalEngTime = 0.0;
    let totalAirTime = 0.0;
    let totalRapTime = 0.0;
    let weightedHex = 0.0;
    let weightedHexSq = 0.0;
    let optHexTime = 0.0;
    let peakMrr = 0.0;
    let minCutF = Infinity;
    let maxCutF = -Infinity;
    let sumCutF = 0.0;
    let nCut = 0;

    for (let i = 0; i < srcSegs.length; i++) {
      const src = srcSegs[i];
      const isAtcDwell =
        src.kind === "rapid" &&
        Number(src.length_mm || 0.0) <= 1e-6 &&
        Number(src.duration_s || 0.0) >= 0.35;
      const dur = isAtcDwell ? Number(src.duration_s) : Math.max(1e-7, Number(durs[i]));
      const spd = Number(speeds[i]);
      const isRapid = src.kind === "rapid";
      const tb = getToolBaseline(src.tool_number || 1);
      const tCutter = tb.cutter;
      const tRpm = Math.max(100.0, Number(tb.res.spindle_rpm || spindleRpmPrimary));
      const toothFreqHz = (tRpm * Math.max(1, tCutter.flutes)) / 60.0;
      const hOptTarget = Math.max(0.004, Number(tb.res.chip_thickness_mm || hOptPrimary));
      const hBandMin = 0.65 * hOptTarget;
      const hBandMax = 1.28 * hOptTarget;

      let engT = 0.0;
      let airT = 0.0;
      let fzMm = 0.0;
      let hexMm = 0.0;
      let mrr = 0.0;

      if (isRapid) {
        totalRapTime += dur;
      } else {
        minCutF = Math.min(minCutF, spd);
        maxCutF = Math.max(maxCutF, spd);
        sumCutF += spd;
        nCut++;

        const eFrac = Math.max(0.0, Math.min(1.0, Number(src.engaged_fraction || (src.engaged ? 1.0 : 0.0))));
        engT = dur * eFrac;
        airT = dur * (1.0 - eFrac);
        totalEngTime += engT;
        totalAirTime += airT;

        fzMm = spd / Math.max(toothFreqHz, 1e-6);
        if (src.engaged && engT > 1e-7) {
          const uAe = Math.max(0.0, Math.min(1.0, Number(src.ae_mm || 0.0) / Math.max(tCutter.diameter_mm, 1e-6)));
          const psi = tb.isHole ? 1.0 : radialChipThinningPsi(uAe);
          hexMm = fzMm * psi;
          weightedHex += hexMm * engT;
          const diffH = hexMm - hOptTarget;
          weightedHexSq += diffH * diffH * engT;
          if (hexMm >= hBandMin && hexMm <= hBandMax) {
            optHexTime += engT;
          }
          mrr = Number(src.removed_mm3 || 0.0) / Math.max(dur, 1e-6);
          if (mrr > peakMrr) peakMrr = mrr;
        }
      }

      newSegs[i] = {
        id: src.id,
        kind: src.kind,
        class: src.class,
        start: src.start,
        end: src.end,
        length_mm: src.length_mm,
        feed_mm_s: Number(spd.toFixed(4)),
        feed_mm_min: Number((spd * 60.0).toFixed(2)),
        t0_s: Number(tCursor.toFixed(5)),
        t1_s: Number((tCursor + dur).toFixed(5)),
        duration_s: Number(dur.toFixed(5)),
        engaged: Boolean(src.engaged),
        engaged_fraction: Number(src.engaged_fraction || 0.0),
        engaged_time_s: Number(engT.toFixed(5)),
        air_cut_time_s: Number(airT.toFixed(5)),
        removed_mm3: Number(src.removed_mm3 || 0.0),
        cum_removed_mm3: Number(src.cum_removed_mm3 || 0.0),
        mrr_mm3_s: Number(mrr.toFixed(3)),
        ap_mm: Number(src.ap_mm || nomDoc),
        ae_mm: Number(src.ae_mm || 0.0),
        ae_pct: Number(src.ae_pct || 0.0),
        phi_eng_deg: Number(src.phi_eng_deg || 0.0),
        fz_mm: Number(fzMm.toFixed(5)),
        hex_mm: Number(hexMm.toFixed(5)),
        tool_number: Number(src.tool_number || 1),
        tool_id: src.tool_id || (tb.rec && tb.rec.tool_id) || "",
        tool_radius_mm: Number(src.tool_radius_mm || (tCutter.diameter_mm * 0.5)),
        entity_type: src.entity_type || tCutter.entity_type || "flat end mill",
        corner_radius_mm: Number(
          src.corner_radius_mm !== undefined ? src.corner_radius_mm : tCutter.corner_radius_mm || 0.0
        ),
        tip_angle_deg: Number(
          src.tip_angle_deg !== undefined
            ? src.tip_angle_deg
            : (tb.rec && tb.rec.tip_angle_deg) || 180.0
        ),
        setup_index: Number(src.setup_index || 1),
        work_offset: src.work_offset || (Number(src.setup_index || 1) === 2 ? "G55" : "G54"),
        op_index: src.op_index !== undefined ? Number(src.op_index) : undefined,
        op_id: src.op_id || "",
        op_kind: src.op_kind || "",
        gcode_line: src.gcode_line !== undefined ? Number(src.gcode_line) : null,
      };
      tCursor += dur;
    }

    const totalT = Math.max(tCursor, 1e-9);
    const denomEng = Math.max(totalEngTime, 1e-9);
    const origEng = origPlan.engagement_summary || {};
    const hBandPrimary = [Number((0.70 * hOptPrimary).toFixed(5)), Number((1.25 * hOptPrimary).toFixed(5))];

    variant.plan = Object.assign({}, origPlan, {
      total_duration_s: Number(tCursor.toFixed(4)),
      min_cut_feed_mm_s: nCut > 0 ? Number(minCutF.toFixed(3)) : 0.0,
      max_cut_feed_mm_s: nCut > 0 ? Number(maxCutF.toFixed(3)) : 0.0,
      mean_cut_feed_mm_s: nCut > 0 ? Number((sumCutF / nCut).toFixed(3)) : 0.0,
      segments: newSegs,
      engagement_summary: Object.assign({}, origEng, {
        total_time_s: Number(tCursor.toFixed(4)),
        engaged_time_s: Number(totalEngTime.toFixed(4)),
        air_cut_time_s: Number(totalAirTime.toFixed(4)),
        rapid_time_s: Number(totalRapTime.toFixed(4)),
        engaged_pct: Number(((100.0 * totalEngTime) / totalT).toFixed(2)),
        air_cut_pct: Number(((100.0 * totalAirTime) / totalT).toFixed(2)),
        rapid_pct: Number(((100.0 * totalRapTime) / totalT).toFixed(2)),
        mean_hex_mm: totalEngTime > 1e-6 ? Number((weightedHex / denomEng).toFixed(5)) : 0.0,
        hex_rmse_mm: totalEngTime > 1e-6 ? Number(Math.sqrt(weightedHexSq / denomEng).toFixed(5)) : 0.0,
        target_hex_mm: Number(hOptPrimary.toFixed(5)),
        optimal_hex_band_mm: hBandPrimary,
        optimal_hex_pct: totalEngTime > 1e-6 ? Number(((100.0 * optHexTime) / denomEng).toFixed(2)) : 0.0,
        peak_inst_mrr_mm3_s: Number(peakMrr.toFixed(2)),
        spindle_rpm: spindleRpmPrimary,
      }),
    });

    if (!variant.origMetrics) {
      variant.origMetrics = Object.assign({}, variant.metrics);
    }
    variant.metrics.cycle_time_s = Number(tCursor.toFixed(3));
    variant.metrics.peak_mrr_mm3_s = Number(peakMrr.toFixed(2));
    variant._liveGcodeCache = null;
  }

  // --- Safe HTML & SVG DOM Helpers (zero innerHTML / eval / inline styles) ---
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined) continue;
        if (k === "className") {
          node.className = String(v);
        } else if (k === "text") {
          node.textContent = String(v);
        } else {
          node.setAttribute(k, String(v));
        }
      }
    }
    if (Array.isArray(children)) {
      for (const child of children) {
        if (child === null || child === undefined) continue;
        if (typeof child === "string") {
          node.appendChild(document.createTextNode(child));
        } else {
          node.appendChild(child);
        }
      }
    }
    return node;
  }

  function svgEl(tag, attrs, children) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined) continue;
        node.setAttribute(k, String(v));
      }
    }
    if (Array.isArray(children)) {
      for (const child of children) {
        if (child) node.appendChild(child);
      }
    }
    return node;
  }

  function showError(msg) {
    const banner = document.getElementById("error-banner");
    if (!banner) return;
    banner.textContent = String(msg);
    banner.classList.remove("hidden");
  }

  function clearError() {
    const banner = document.getElementById("error-banner");
    if (!banner) return;
    banner.textContent = "";
    banner.classList.add("hidden");
  }

  // --- Minimal 4x4 Column-Major Matrix Utilities (Z-up G54 Frame) ---
  function mat4Identity() {
    const m = new Float32Array(16);
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
    return m;
  }

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  function mat4Perspective(fovRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovRad * 0.5);
    const nf = 1.0 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / Math.max(aspect, 1e-4);
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1.0;
    out[14] = 2.0 * far * near * nf;
    return out;
  }

  function mat4LookAt(eye, target, up) {
    let zx = eye[0] - target[0];
    let zy = eye[1] - target[1];
    let zz = eye[2] - target[2];
    const zLen = Math.hypot(zx, zy, zz) || 1.0;
    zx /= zLen; zy /= zLen; zz /= zLen;

    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    const xLen = Math.hypot(xx, xy, xz);
    if (xLen < 1e-6) {
      xx = 1.0; xy = 0.0; xz = 0.0;
    } else {
      xx /= xLen; xy /= xLen; xz /= xLen;
    }

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    const out = mat4Identity();
    out[0] = xx; out[4] = xy; out[8] = xz;
    out[1] = yx; out[5] = yy; out[9] = yz;
    out[2] = zx; out[6] = zy; out[10] = zz;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    return out;
  }

  function mat4Translation(tx, ty, tz) {
    const out = mat4Identity();
    out[12] = tx; out[13] = ty; out[14] = tz;
    return out;
  }

  // Proper SO(3) 180-deg flip around X axis through zMid = 0.5 * zFlipSum:
  // x' = x, y' = -y, z' = zFlipSum - z (det = +1, preserves outward normals & triangle winding!)
  function mat4FlipX180(zFlipSum) {
    const out = mat4Identity();
    out[0] = 1.0;
    out[5] = -1.0;
    out[10] = -1.0;
    out[14] = Number(zFlipSum || 0.0);
    return out;
  }

  // --- Validated Binary STL Parser ---
  function parseBinaryStl(buffer, tintRgb) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 84) {
      throw new Error("Invalid binary STL payload (< 84 bytes).");
    }
    const view = new DataView(buffer);
    const triCount = view.getUint32(80, true);
    if (triCount > MAX_STL_TRIANGLES || 84 + triCount * 50 !== buffer.byteLength) {
      throw new Error("STL facet count does not match byte length.");
    }
    const positions = new Float32Array(triCount * 9);
    const normals = new Float32Array(triCount * 9);
    const colors = new Float32Array(triCount * 9);
    const cr = tintRgb ? tintRgb[0] : 0.72;
    const cg = tintRgb ? tintRgb[1] : 0.78;
    const cb = tintRgb ? tintRgb[2] : 0.85;

    let offset = 84;
    for (let i = 0; i < triCount; i++) {
      let nx = view.getFloat32(offset, true);
      let ny = view.getFloat32(offset + 4, true);
      let nz = view.getFloat32(offset + 8, true);

      const v0x = view.getFloat32(offset + 12, true);
      const v0y = view.getFloat32(offset + 16, true);
      const v0z = view.getFloat32(offset + 20, true);
      const v1x = view.getFloat32(offset + 24, true);
      const v1y = view.getFloat32(offset + 28, true);
      const v1z = view.getFloat32(offset + 32, true);
      const v2x = view.getFloat32(offset + 36, true);
      const v2y = view.getFloat32(offset + 40, true);
      const v2z = view.getFloat32(offset + 44, true);
      offset += 50;

      let nLen = Math.hypot(nx, ny, nz);
      if (!Number.isFinite(nLen) || nLen < 1e-6) {
        const ax = v1x - v0x, ay = v1y - v0y, az = v1z - v0z;
        const bx = v2x - v0x, by = v2y - v0y, bz = v2z - v0z;
        nx = ay * bz - az * by;
        ny = az * bx - ax * bz;
        nz = ax * by - ay * bx;
        nLen = Math.hypot(nx, ny, nz) || 1.0;
      }
      nx /= nLen; ny /= nLen; nz /= nLen;

      const base = i * 9;
      positions[base] = v0x; positions[base + 1] = v0y; positions[base + 2] = v0z;
      positions[base + 3] = v1x; positions[base + 4] = v1y; positions[base + 5] = v1z;
      positions[base + 6] = v2x; positions[base + 7] = v2y; positions[base + 8] = v2z;

      normals[base] = nx; normals[base + 1] = ny; normals[base + 2] = nz;
      normals[base + 3] = nx; normals[base + 4] = ny; normals[base + 5] = nz;
      normals[base + 6] = nx; normals[base + 7] = ny; normals[base + 8] = nz;

      colors[base] = cr; colors[base + 1] = cg; colors[base + 2] = cb;
      colors[base + 3] = cr; colors[base + 4] = cg; colors[base + 5] = cb;
      colors[base + 6] = cr; colors[base + 7] = cg; colors[base + 8] = cb;
    }
    return { positions, normals, colors, vertexCount: triCount * 3 };
  }

  // --- Clean-Room Real-Time 2.5D Dual-Sided (Op1 G54 + Op2 G55) Profile-Aware Dexel Stock Simulator ---
  function classifyTipProfile(seg, caseToolsTable) {
    const tNum = Number((seg && seg.tool_number) || 1);
    const tRec = (caseToolsTable && (caseToolsTable[String(tNum)] || caseToolsTable[tNum])) || null;
    const etype = String((seg && seg.entity_type) || (tRec && tRec.entity_type) || "flat end mill").toLowerCase();
    const r = Math.max(0.15, Number((seg && seg.tool_radius_mm) || (tRec && tRec.cutter_radius_mm) || 2.5));
    const rc = Math.max(
      0.0,
      Number(
        seg && seg.corner_radius_mm !== undefined
          ? seg.corner_radius_mm
          : (tRec && tRec.corner_radius_mm) || 0.0
      )
    );
    const tipDeg = Number(
      seg && seg.tip_angle_deg !== undefined
        ? seg.tip_angle_deg
        : (tRec && tRec.tip_angle_deg) || 180.0
    );

    if (etype === "probe") {
      return { kind: -1, r, rc: 0.0, cotHalf: 0.0 }; // touch probe never removes metal
    }
    if (etype === "ball end mill" || etype === "lollipop mill" || rc >= 0.92 * r) {
      return { kind: 1, r, rc: r, cotHalf: 0.0 }; // hemisphere ball tip
    }
    if (
      etype === "drill" ||
      etype === "spot drill" ||
      etype === "center drill" ||
      etype === "chamfer mill" ||
      etype === "counter sink" ||
      (tipDeg > 15.0 && tipDeg < 172.0)
    ) {
      const halfRad = (Math.max(25.0, Math.min(165.0, tipDeg || 118.0)) * Math.PI) / 360.0;
      return { kind: 3, r, rc: 0.0, cotHalf: 1.0 / Math.max(0.15, Math.tan(halfRad)) };
    }
    if (rc > 0.05) {
      return { kind: 2, r, rc: Math.min(rc, r), cotHalf: 0.0 }; // bull-nose corner torus
    }
    return { kind: 0, r, rc: 0.0, cotHalf: 0.0 }; // flat end mill
  }

  function tipProfileDeltaZ(profile, dPerp2) {
    if (profile.kind === 0) return 0.0;
    if (profile.kind === 1) {
      const r2 = profile.r * profile.r;
      return profile.r - Math.sqrt(Math.max(0.0, r2 - dPerp2));
    }
    const d = Math.sqrt(Math.max(0.0, dPerp2));
    if (profile.kind === 3) {
      return d * profile.cotHalf;
    }
    const flatCore = profile.r - profile.rc;
    if (d <= flatCore) return 0.0;
    const dr = d - flatCore;
    return profile.rc - Math.sqrt(Math.max(0.0, profile.rc * profile.rc - dr * dr));
  }

  class DexelStockSim {
    constructor(caseData, pitchMm) {
      const basePitch = pitchMm || DEXEL_PITCH_MM;
      this.smin = caseData.stock_min_mm.map(Number);
      this.smax = caseData.stock_max_mm.map(Number);
      this.zFlipSum = this.smin[2] + this.smax[2];
      this.toolsTable = caseData.tools_table || null;
      this.rCut = Number((caseData.tool && caseData.tool.cutter_radius_mm) || 2.5);
      this.r2 = this.rCut * this.rCut;

      const lx = Math.max(2.0, this.smax[0] - this.smin[0]);
      const ly = Math.max(2.0, this.smax[1] - this.smin[1]);
      // Adaptive pitch cap so large parts (e.g. 8" Titan-M10) stay < 2 ms per carve frame
      this.pitch = Math.max(basePitch, Math.max(lx, ly) / 195.0);
      this.nx = Math.max(8, Math.min(210, Math.round(lx / this.pitch)));
      this.ny = Math.max(8, Math.min(210, Math.round(ly / this.pitch)));
      this.dx = lx / this.nx;
      this.dy = ly / this.ny;

      const count = this.nx * this.ny;
      this.xs = new Float32Array(this.nx);
      this.ys = new Float32Array(this.ny);
      for (let i = 0; i < this.nx; i++) this.xs[i] = this.smin[0] + (i + 0.5) * this.dx;
      for (let j = 0; j < this.ny; j++) this.ys[j] = this.smin[1] + (j + 0.5) * this.dy;

      this.zTarget = new Float32Array(count);
      this._buildAnalyticTarget(caseData);

      this.zWork = new Float32Array(count);
      this.zBot = new Float32Array(count);
      this.lastCutSeg = new Int32Array(count);
      this.hasOp2 = false;

      const maxTris = (6 * count + 4 * (this.nx + this.ny) + 12) * 2;
      this.posScratch = new Float32Array(maxTris * 9);
      this.nrmScratch = new Float32Array(maxTris * 9);
      this.colScratch = new Float32Array(maxTris * 9);

      this.segments = [];
      this.keyframes = [];
      this.keyframesBot = [];
      this.keyframeCutSeg = [];
      this.lastCarveKey = "";
    }

    _buildAnalyticTarget(caseData) {
      const [bx0, by0, bz0] = this.smin;
      const [bx1, by1, bz1] = this.smax;
      const pmin = (caseData.pocket_min_mm || [bx0, by0, bz1]).map(Number);
      const pmax = (caseData.pocket_max_mm || [bx1, by1, bz1]).map(Number);
      const rc = Number(caseData.corner_radius_mm || 0.0);
      const islands = Array.isArray(caseData.islands_mm) ? caseData.islands_mm : [];

      const xWalledLo = pmin[0] > bx0 + 1e-6;
      const xWalledHi = pmax[0] < bx1 - 1e-6;
      const yWalledLo = pmin[1] > by0 + 1e-6;
      const yWalledHi = pmax[1] < by1 - 1e-6;

      for (let ix = 0; ix < this.nx; ix++) {
        const px = this.xs[ix];
        for (let iy = 0; iy < this.ny; iy++) {
          const py = this.ys[iy];
          let zt = bz1;

          if (px >= pmin[0] && px <= pmax[0] && py >= pmin[1] && py <= pmax[1]) {
            let inCornerFillet = false;
            if (rc > 1e-5) {
              if (xWalledLo && yWalledLo && px < pmin[0] + rc && py < pmin[1] + rc) {
                if (Math.hypot(px - (pmin[0] + rc), py - (pmin[1] + rc)) > rc) inCornerFillet = true;
              }
              if (xWalledHi && yWalledLo && px > pmax[0] - rc && py < pmin[1] + rc) {
                if (Math.hypot(px - (pmax[0] - rc), py - (pmin[1] + rc)) > rc) inCornerFillet = true;
              }
              if (xWalledLo && yWalledHi && px < pmin[0] + rc && py > pmax[1] - rc) {
                if (Math.hypot(px - (pmin[0] + rc), py - (pmax[1] - rc)) > rc) inCornerFillet = true;
              }
              if (xWalledHi && yWalledHi && px > pmax[0] - rc && py > pmax[1] - rc) {
                if (Math.hypot(px - (pmax[0] - rc), py - (pmax[1] - rc)) > rc) inCornerFillet = true;
              }
            }
            if (!inCornerFillet) {
              zt = Math.max(bz0, pmin[2]);
            }
          }

          for (let k = 0; k < islands.length; k++) {
            const lo = islands[k][0];
            const hi = islands[k][1];
            if (px >= lo[0] && px <= hi[0] && py >= lo[1] && py <= hi[1]) {
              zt = Math.max(zt, Number(hi[2]));
            }
          }
          this.zTarget[ix * this.ny + iy] = zt;
        }
      }
    }

    // Rasterize actual parsed target STL triangles so Stock-to-Model Deviation is exact on arbitrary STEP/STL parts
    setTargetFromMesh(parsedMesh) {
      if (!parsedMesh || !parsedMesh.positions || parsedMesh.vertexCount < 3) return;
      const p = parsedMesh.positions;
      const [bx0, by0, bz0] = this.smin;
      this.zTarget.fill(bz0);
      const nTris = Math.floor(parsedMesh.vertexCount / 3);
      for (let t = 0; t < nTris; t++) {
        const b = t * 9;
        const x0 = p[b], y0 = p[b + 1], z0 = p[b + 2];
        const x1 = p[b + 3], y1 = p[b + 4], z1 = p[b + 5];
        const x2 = p[b + 6], y2 = p[b + 7], z2 = p[b + 8];
        const denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (denom <= 1e-9) continue; // ignore downward / vertical faces
        const ix0 = Math.max(0, Math.floor((Math.min(x0, x1, x2) - bx0) / this.dx));
        const ix1 = Math.min(this.nx - 1, Math.ceil((Math.max(x0, x1, x2) - bx0) / this.dx));
        const iy0 = Math.max(0, Math.floor((Math.min(y0, y1, y2) - by0) / this.dy));
        const iy1 = Math.min(this.ny - 1, Math.ceil((Math.max(y0, y1, y2) - by0) / this.dy));
        const invD = 1.0 / denom;
        for (let ix = ix0; ix <= ix1; ix++) {
          const px = this.xs[ix];
          const rowOff = ix * this.ny;
          for (let iy = iy0; iy <= iy1; iy++) {
            const py = this.ys[iy];
            const w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * invD;
            const w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * invD;
            const w2 = 1.0 - w0 - w1;
            if (w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4) {
              const zt = w0 * z0 + w1 * z1 + w2 * z2;
              if (zt > this.zTarget[rowOff + iy]) {
                this.zTarget[rowOff + iy] = zt;
              }
            }
          }
        }
      }
      this.lastCarveKey = "";
    }

    setSegments(segments) {
      this.segments = Array.isArray(segments) ? segments : [];
      this.keyframes = [];
      this.keyframesBot = [];
      this.keyframeCutSeg = [];
      this.hasOp2 = false;

      const zCur = new Float32Array(this.nx * this.ny);
      zCur.fill(this.smax[2]);
      const zBotCur = new Float32Array(this.nx * this.ny);
      zBotCur.fill(this.smin[2]);
      const cutSeg = new Int32Array(this.nx * this.ny);
      cutSeg.fill(-999);

      this.keyframes.push(zCur.slice());
      this.keyframesBot.push(zBotCur.slice());
      this.keyframeCutSeg.push(cutSeg.slice());

      for (let k = 0; k < this.segments.length; k++) {
        const s = this.segments[k];
        if (Number(s.setup_index || 1) === 2) {
          this.hasOp2 = true;
        }
        if (s.kind !== "rapid") {
          this._sweepSegment(zCur, zBotCur, cutSeg, k, s, 1.0);
        }
        if ((k + 1) % KEYFRAME_STRIDE === 0) {
          this.keyframes.push(zCur.slice());
          this.keyframesBot.push(zBotCur.slice());
          this.keyframeCutSeg.push(cutSeg.slice());
        }
      }
      this.lastCarveKey = "";
    }

    _sweepSegment(zTopGrid, zBotGrid, cutSeg, segIdx, seg, alpha) {
      const profile = classifyTipProfile(seg, this.toolsTable);
      if (profile.kind < 0) return; // spindle touch probe
      const ax = seg.start[0];
      const ay = seg.start[1];
      const az = seg.start[2];
      const bx = alpha >= 0.99999 ? seg.end[0] : ax + (seg.end[0] - ax) * alpha;
      const by = alpha >= 0.99999 ? seg.end[1] : ay + (seg.end[1] - ay) * alpha;
      const bz = alpha >= 0.99999 ? seg.end[2] : az + (seg.end[2] - az) * alpha;
      const isOp2 = Number(seg.setup_index || 1) === 2;
      if (isOp2) {
        // In G55 (180-deg X flip), x_g54 = x_g55, y_g54 = -y_g55, z_g54_bot = zFlipSum - z_g55_cut
        this._sweepLinearOp2(zBotGrid, cutSeg, segIdx, ax, -ay, az, bx, -by, bz, profile);
      } else {
        this._sweepLinearOp1(zTopGrid, cutSeg, segIdx, ax, ay, az, bx, by, bz, profile);
      }
    }

    _sweepLinearOp1(zGrid, cutSeg, segIdx, ax, ay, az, bx, by, bz, profile) {
      const zMinSeg = Math.min(az, bz);
      if (zMinSeg >= this.smax[2] - 1e-6) return;

      const r = profile.r;
      const r2Local = r * r;
      const ix0 = Math.max(0, Math.floor((Math.min(ax, bx) - r - this.smin[0]) / this.dx));
      const ix1 = Math.min(this.nx - 1, Math.ceil((Math.max(ax, bx) + r - this.smin[0]) / this.dx));
      const iy0 = Math.max(0, Math.floor((Math.min(ay, by) - r - this.smin[1]) / this.dy));
      const iy1 = Math.min(this.ny - 1, Math.ceil((Math.max(ay, by) + r - this.smin[1]) / this.dy));
      if (ix0 > ix1 || iy0 > iy1) return;

      const dx = bx - ax;
      const dy = by - ay;
      const dz = bz - az;
      const lxy2 = dx * dx + dy * dy;
      const zFloor = this.smin[2];

      if (lxy2 <= 1e-10) {
        for (let ix = ix0; ix <= ix1; ix++) {
          const dpx = this.xs[ix] - ax;
          const dpx2 = dpx * dpx;
          const rowOff = ix * this.ny;
          for (let iy = iy0; iy <= iy1; iy++) {
            const dpy = this.ys[iy] - ay;
            const d2 = dpx2 + dpy * dpy;
            if (d2 <= r2Local) {
              const zCut = Math.max(zFloor, zMinSeg + tipProfileDeltaZ(profile, d2));
              const idx = rowOff + iy;
              if (zCut < zGrid[idx] - 1e-6) {
                zGrid[idx] = zCut;
                cutSeg[idx] = segIdx;
              }
            }
          }
        }
        return;
      }

      const invLxy2 = 1.0 / lxy2;
      for (let ix = ix0; ix <= ix1; ix++) {
        const px = this.xs[ix] - ax;
        const rowOff = ix * this.ny;
        for (let iy = iy0; iy <= iy1; iy++) {
          const py = this.ys[iy] - ay;
          const uc = (px * dx + py * dy) * invLxy2;
          const perpX = px - uc * dx;
          const perpY = py - uc * dy;
          const dPerp2 = perpX * perpX + perpY * perpY;
          if (dPerp2 > r2Local) continue;

          const du = Math.sqrt(Math.max(0.0, r2Local - dPerp2) * invLxy2);
          const uLo = Math.max(0.0, uc - du);
          const uHi = Math.min(1.0, uc + du);
          if (uLo > uHi) continue;

          const uNear = Math.max(0.0, Math.min(1.0, uc));
          const ux = px - uNear * dx;
          const uy = py - uNear * dy;
          const d2Min = ux * ux + uy * uy;
          const uBest = dz < 0.0 ? uHi : uLo;
          const zCut = Math.max(zFloor, az + uBest * dz + tipProfileDeltaZ(profile, d2Min));
          const idx = rowOff + iy;
          if (zCut < zGrid[idx] - 1e-6) {
            zGrid[idx] = zCut;
            cutSeg[idx] = segIdx;
          }
        }
      }
    }

    _sweepLinearOp2(zBotGrid, cutSeg, segIdx, ax, ayG54, azG55, bx, byG54, bzG55, profile) {
      const zMinSeg = Math.min(azG55, bzG55);
      if (zMinSeg >= this.smax[2] - 1e-6) return;

      const r = profile.r;
      const r2Local = r * r;
      const ix0 = Math.max(0, Math.floor((Math.min(ax, bx) - r - this.smin[0]) / this.dx));
      const ix1 = Math.min(this.nx - 1, Math.ceil((Math.max(ax, bx) + r - this.smin[0]) / this.dx));
      const iy0 = Math.max(0, Math.floor((Math.min(ayG54, byG54) - r - this.smin[1]) / this.dy));
      const iy1 = Math.min(this.ny - 1, Math.ceil((Math.max(ayG54, byG54) + r - this.smin[1]) / this.dy));
      if (ix0 > ix1 || iy0 > iy1) return;

      const dx = bx - ax;
      const dy = byG54 - ayG54;
      const dz = bzG55 - azG55;
      const lxy2 = dx * dx + dy * dy;
      const zCeil = this.smax[2];
      const zSum = this.zFlipSum;

      if (lxy2 <= 1e-10) {
        for (let ix = ix0; ix <= ix1; ix++) {
          const dpx = this.xs[ix] - ax;
          const dpx2 = dpx * dpx;
          const rowOff = ix * this.ny;
          for (let iy = iy0; iy <= iy1; iy++) {
            const dpy = this.ys[iy] - ayG54;
            const d2 = dpx2 + dpy * dpy;
            if (d2 <= r2Local) {
              const zCutG55 = zMinSeg + tipProfileDeltaZ(profile, d2);
              const zBotNew = Math.min(zCeil, zSum - zCutG55);
              const idx = rowOff + iy;
              if (zBotNew > zBotGrid[idx] + 1e-6) {
                zBotGrid[idx] = zBotNew;
                cutSeg[idx] = segIdx;
              }
            }
          }
        }
        return;
      }

      const invLxy2 = 1.0 / lxy2;
      for (let ix = ix0; ix <= ix1; ix++) {
        const px = this.xs[ix] - ax;
        const rowOff = ix * this.ny;
        for (let iy = iy0; iy <= iy1; iy++) {
          const py = this.ys[iy] - ayG54;
          const uc = (px * dx + py * dy) * invLxy2;
          const perpX = px - uc * dx;
          const perpY = py - uc * dy;
          const dPerp2 = perpX * perpX + perpY * perpY;
          if (dPerp2 > r2Local) continue;

          const du = Math.sqrt(Math.max(0.0, r2Local - dPerp2) * invLxy2);
          const uLo = Math.max(0.0, uc - du);
          const uHi = Math.min(1.0, uc + du);
          if (uLo > uHi) continue;

          const uNear = Math.max(0.0, Math.min(1.0, uc));
          const ux = px - uNear * dx;
          const uy = py - uNear * dy;
          const d2Min = ux * ux + uy * uy;
          const uBest = dz < 0.0 ? uHi : uLo;
          const zCutG55 = azG55 + uBest * dz + tipProfileDeltaZ(profile, d2Min);
          const zBotNew = Math.min(zCeil, zSum - zCutG55);
          const idx = rowOff + iy;
          if (zBotNew > zBotGrid[idx] + 1e-6) {
            zBotGrid[idx] = zBotNew;
            cutSeg[idx] = segIdx;
          }
        }
      }
    }

    carveAndTriangulate(activeSegIdx, alpha, stockMode, paletteKey) {
      const clampedSeg = Math.max(-1, Math.min(activeSegIdx, this.segments.length - 1));
      const quantAlpha = Math.round(Math.max(0.0, Math.min(1.0, alpha)) * 40);
      const cacheKey = clampedSeg + ":" + quantAlpha + ":" + stockMode + ":" + paletteKey;
      if (cacheKey === this.lastCarveKey) {
        return null;
      }
      this.lastCarveKey = cacheKey;

      const completedCount = clampedSeg < 0 ? 0 : clampedSeg;
      const kfIndex = Math.min(
        this.keyframes.length - 1,
        Math.floor(completedCount / KEYFRAME_STRIDE)
      );
      this.zWork.set(this.keyframes[kfIndex]);
      this.zBot.set(this.keyframesBot[kfIndex]);
      this.lastCutSeg.set(this.keyframeCutSeg[kfIndex]);

      const replayStart = kfIndex * KEYFRAME_STRIDE;
      for (let k = replayStart; k < completedCount; k++) {
        const s = this.segments[k];
        if (s.kind !== "rapid") {
          this._sweepSegment(this.zWork, this.zBot, this.lastCutSeg, k, s, 1.0);
        }
      }

      if (clampedSeg >= 0 && alpha > 1e-4) {
        const cur = this.segments[clampedSeg];
        if (cur && cur.kind !== "rapid") {
          this._sweepSegment(this.zWork, this.zBot, this.lastCutSeg, clampedSeg, cur, alpha);
        }
      }

      return this._triangulate(clampedSeg, stockMode, paletteKey);
    }

    _cellTopColor(idx, zVal, activeSegIdx, stockMode, pal) {
      if (stockMode === "live_deviation") {
        const diff = zVal - this.zTarget[idx];
        if (diff < -0.08) return [0.94, 0.24, 0.24];
        if (diff <= 0.16) return [0.19, 0.72, 0.30];
        const u = Math.min(1.0, (diff - 0.16) / 4.5);
        return [0.16 * (1 - u) + 0.14 * u, 0.56 * (1 - u) + 0.72 * u, 0.92];
      }
      if (stockMode === "live_glass") {
        return pal.skin;
      }
      if (zVal >= this.smax[2] - 1e-4) {
        return pal.skin;
      }
      const age = activeSegIdx - this.lastCutSeg[idx];
      if (age >= 0 && age <= 3) {
        const glow = 1.0 - age / 4.0;
        return [
          pal.alloy[0] * (1 - glow * 0.45) + 0.98 * glow * 0.45,
          pal.alloy[1] * (1 - glow * 0.45) + 0.68 * glow * 0.45,
          pal.alloy[2] * (1 - glow * 0.45) + 0.22 * glow * 0.45,
        ];
      }
      return pal.alloy;
    }

    _cellBotColor(idx, zBotVal, activeSegIdx, stockMode, pal) {
      if (stockMode === "live_glass" || zBotVal <= this.smin[2] + 1e-4) {
        return pal.skin;
      }
      const age = activeSegIdx - this.lastCutSeg[idx];
      if (age >= 0 && age <= 3) {
        const glow = 1.0 - age / 4.0;
        return [
          pal.alloy[0] * (1 - glow * 0.45) + 0.98 * glow * 0.45,
          pal.alloy[1] * (1 - glow * 0.45) + 0.68 * glow * 0.45,
          pal.alloy[2] * (1 - glow * 0.45) + 0.22 * glow * 0.45,
        ];
      }
      return pal.alloy;
    }

    _triangulate(activeSegIdx, stockMode, paletteKey) {
      const pal = STOCK_PALETTES[paletteKey] || STOCK_PALETTES.green;
      const pos = this.posScratch;
      const nrm = this.nrmScratch;
      const col = this.colScratch;
      let o = 0;

      function emitQuad(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, nx, ny, nz, rgb) {
        const r = rgb[0], g = rgb[1], b = rgb[2];
        pos[o] = ax; pos[o + 1] = ay; pos[o + 2] = az;
        pos[o + 3] = bx; pos[o + 4] = by; pos[o + 5] = bz;
        pos[o + 6] = cx; pos[o + 7] = cy; pos[o + 8] = cz;
        pos[o + 9] = ax; pos[o + 10] = ay; pos[o + 11] = az;
        pos[o + 12] = cx; pos[o + 13] = cy; pos[o + 14] = cz;
        pos[o + 15] = dx; pos[o + 16] = dy; pos[o + 17] = dz;

        for (let k = 0; k < 18; k += 3) {
          nrm[o + k] = nx; nrm[o + k + 1] = ny; nrm[o + k + 2] = nz;
          col[o + k] = r; col[o + k + 1] = g; col[o + k + 2] = b;
        }
        o += 18;
      }

      const [bx0, by0, bz0] = this.smin;
      const [bx1, by1] = this.smax;
      const dx = this.dx;
      const dy = this.dy;
      const nx = this.nx;
      const ny = this.ny;
      const z = this.zWork;
      const zb = this.zBot;
      const wallCutRgb = stockMode === "live_glass" ? pal.skin : pal.wall;

      for (let ix = 0; ix < nx; ix++) {
        const x0 = bx0 + ix * dx;
        const x1 = x0 + dx;
        const rowOff = ix * ny;
        for (let iy = 0; iy < ny; iy++) {
          const idx = rowOff + iy;
          const zTop = z[idx];
          const zLow = zb[idx];
          if (zTop <= zLow + 1e-4) continue; // through-cut / breakthrough hole!
          const y0 = by0 + iy * dy;
          const y1 = y0 + dy;
          const cTop = this._cellTopColor(idx, zTop, activeSegIdx, stockMode, pal);
          emitQuad(x0, y0, zTop, x1, y0, zTop, x1, y1, zTop, x0, y1, zTop, 0, 0, 1, cTop);
          if (this.hasOp2) {
            const cBot = this._cellBotColor(idx, zLow, activeSegIdx, stockMode, pal);
            emitQuad(x0, y0, zLow, x0, y1, zLow, x1, y1, zLow, x1, y0, zLow, 0, 0, -1, cBot);
          }
        }
      }

      for (let iy = 0; iy < ny; iy++) {
        const y0 = by0 + iy * dy;
        const y1 = y0 + dy;
        const zLeft = z[iy], zbLeft = zb[iy];
        if (zLeft > zbLeft + 1e-4) {
          emitQuad(bx0, y0, zbLeft, bx0, y0, zLeft, bx0, y1, zLeft, bx0, y1, zbLeft, -1, 0, 0, pal.skin);
        }
        const idxR = (nx - 1) * ny + iy;
        const zRight = z[idxR], zbRight = zb[idxR];
        if (zRight > zbRight + 1e-4) {
          emitQuad(bx1, y0, zbRight, bx1, y1, zbRight, bx1, y1, zRight, bx1, y0, zRight, 1, 0, 0, pal.skin);
        }
      }

      for (let ix = 0; ix < nx - 1; ix++) {
        const xEdge = bx0 + (ix + 1) * dx;
        const offA = ix * ny;
        const offB = (ix + 1) * ny;
        for (let iy = 0; iy < ny; iy++) {
          const zA = z[offA + iy], zbA = zb[offA + iy];
          const zB = z[offB + iy], zbB = zb[offB + iy];
          const solidA = zA > zbA + 1e-4;
          const solidB = zB > zbB + 1e-4;
          if (!solidA && !solidB) continue;
          const y0 = by0 + iy * dy;
          const y1 = y0 + dy;
          if (solidA && !solidB) {
            emitQuad(xEdge, y0, zbA, xEdge, y1, zbA, xEdge, y1, zA, xEdge, y0, zA, 1, 0, 0, wallCutRgb);
          } else if (!solidA && solidB) {
            emitQuad(xEdge, y0, zB, xEdge, y0, zbB, xEdge, y1, zbB, xEdge, y1, zB, -1, 0, 0, wallCutRgb);
          } else {
            if (Math.abs(zA - zB) > 1e-4) {
              if (zA > zB) {
                const cStep = stockMode === "live_deviation" ? this._cellTopColor(offA + iy, zA, activeSegIdx, stockMode, pal) : wallCutRgb;
                emitQuad(xEdge, y0, zB, xEdge, y1, zB, xEdge, y1, zA, xEdge, y0, zA, 1, 0, 0, cStep);
              } else {
                const cStep = stockMode === "live_deviation" ? this._cellTopColor(offB + iy, zB, activeSegIdx, stockMode, pal) : wallCutRgb;
                emitQuad(xEdge, y0, zA, xEdge, y0, zB, xEdge, y1, zB, xEdge, y1, zA, -1, 0, 0, cStep);
              }
            }
            if (this.hasOp2 && Math.abs(zbA - zbB) > 1e-4) {
              if (zbA < zbB) {
                emitQuad(xEdge, y0, zbA, xEdge, y1, zbA, xEdge, y1, zbB, xEdge, y0, zbB, 1, 0, 0, wallCutRgb);
              } else {
                emitQuad(xEdge, y0, zbB, xEdge, y0, zbA, xEdge, y1, zbA, xEdge, y1, zbB, -1, 0, 0, wallCutRgb);
              }
            }
          }
        }
      }

      for (let ix = 0; ix < nx; ix++) {
        const x0 = bx0 + ix * dx;
        const x1 = x0 + dx;
        const rowOff = ix * ny;
        const zFront = z[rowOff], zbFront = zb[rowOff];
        if (zFront > zbFront + 1e-4) {
          emitQuad(x0, by0, zbFront, x1, by0, zbFront, x1, by0, zFront, x0, by0, zFront, 0, -1, 0, pal.skin);
        }
        const idxBk = rowOff + ny - 1;
        const zBack = z[idxBk], zbBack = zb[idxBk];
        if (zBack > zbBack + 1e-4) {
          emitQuad(x0, by1, zbBack, x0, by1, zBack, x1, by1, zBack, x1, by1, zbBack, 0, 1, 0, pal.skin);
        }

        for (let iy = 0; iy < ny - 1; iy++) {
          const zA = z[rowOff + iy], zbA = zb[rowOff + iy];
          const zB = z[rowOff + iy + 1], zbB = zb[rowOff + iy + 1];
          const solidA = zA > zbA + 1e-4;
          const solidB = zB > zbB + 1e-4;
          if (!solidA && !solidB) continue;
          const yEdge = by0 + (iy + 1) * dy;
          if (solidA && !solidB) {
            emitQuad(x0, yEdge, zbA, x0, yEdge, zA, x1, yEdge, zA, x1, yEdge, zbA, 0, 1, 0, wallCutRgb);
          } else if (!solidA && solidB) {
            emitQuad(x0, yEdge, zB, x1, yEdge, zB, x1, yEdge, zbB, x0, yEdge, zbB, 0, -1, 0, wallCutRgb);
          } else {
            if (Math.abs(zA - zB) > 1e-4) {
              if (zA > zB) {
                const cStep = stockMode === "live_deviation" ? this._cellTopColor(rowOff + iy, zA, activeSegIdx, stockMode, pal) : wallCutRgb;
                emitQuad(x0, yEdge, zB, x0, yEdge, zA, x1, yEdge, zA, x1, yEdge, zB, 0, 1, 0, cStep);
              } else {
                const cStep = stockMode === "live_deviation" ? this._cellTopColor(rowOff + iy + 1, zB, activeSegIdx, stockMode, pal) : wallCutRgb;
                emitQuad(x0, yEdge, zA, x1, yEdge, zA, x1, yEdge, zB, x0, yEdge, zB, 0, -1, 0, cStep);
              }
            }
            if (this.hasOp2 && Math.abs(zbA - zbB) > 1e-4) {
              if (zbA < zbB) {
                emitQuad(x0, yEdge, zbA, x0, yEdge, zbB, x1, yEdge, zbB, x1, yEdge, zbA, 0, 1, 0, wallCutRgb);
              } else {
                emitQuad(x0, yEdge, zbB, x1, yEdge, zbB, x1, yEdge, zbA, x0, yEdge, zbA, 0, -1, 0, wallCutRgb);
              }
            }
          }
        }
      }

      if (!this.hasOp2) {
        emitQuad(bx0, by0, bz0, bx0, by1, bz0, bx1, by1, bz0, bx1, by0, bz0, 0, 0, -1, pal.skin);
      }

      return {
        positions: pos.subarray(0, o),
        normals: nrm.subarray(0, o),
        colors: col.subarray(0, o),
        vertexCount: o / 3,
      };
    }
  }

  // --- Tool-Library-Linked 3D Cutter + Stepped Holder Mesh Builder (tip at local (0, 0, 0)) ---
  function getCoatingColors(toolMaterialId, entityType, flutes) {
    const mat = String(toolMaterialId || "carbide_coated").toLowerCase();
    const et = String(entityType || "flat end mill").toLowerCase();
    if (mat === "carbide_uncoated") {
      return { land: [0.82, 0.86, 0.91], gullet: [0.54, 0.59, 0.66] };
    }
    if (mat === "cobalt_hss") {
      return { land: [0.80, 0.67, 0.42], gullet: [0.50, 0.41, 0.25] };
    }
    if (mat === "hss") {
      return { land: [0.40, 0.45, 0.52], gullet: [0.23, 0.27, 0.32] };
    }
    if (et === "drill" || et === "chamfer mill" || et === "counter sink" || Number(flutes || 3) >= 4) {
      // TiAlN violet-bronze PVD coating
      return { land: [0.64, 0.54, 0.72], gullet: [0.38, 0.31, 0.46] };
    }
    // ZrN champagne-gold aluminum coating for 2-3FL carbide mills
    return { land: [0.96, 0.78, 0.25], gullet: [0.64, 0.50, 0.14] };
  }

  function buildToolAssemblyMesh(toolSpec, toolMaterialOverride) {
    const spec = toolSpec || {};
    const etype = String(spec.entity_type || "flat end mill").toLowerCase();
    const dMm = Math.max(0.6, Number(spec.diameter_mm || (spec.cutter_radius_mm ? 2.0 * spec.cutter_radius_mm : 5.0)));
    const rCut = 0.5 * dMm;
    const fluteH = Math.max(1.2, Number(spec.flute_length_mm || 2.2 * dMm));
    const rShank = Math.max(0.4, Number(spec.shank_radius_mm || (spec.shank_diameter_mm ? 0.5 * spec.shank_diameter_mm : rCut)));
    const holderBot = Math.max(Number(spec.stickout_mm || spec.holder_bottom_mm || (fluteH + 4.0)), fluteH + 1.5);
    const rHolder = Math.max(rCut * 0.55 + 4.0, rShank + 3.5, Math.min(28.0, Number(spec.holder_radius_mm || 15.0)));
    const holderH = Math.min(Math.max(18.0, Number(spec.holder_height_mm || 28.0)), 32.0);
    const flutes = Math.max(1, Math.min(8, Math.round(Number(spec.flutes || 3))));
    const helixRad = (Number(spec.helix_deg !== undefined ? spec.helix_deg : 35.0) * Math.PI) / 180.0;
    const rc = Math.max(0.0, Math.min(rCut, Number(spec.corner_radius_mm || 0.0)));
    const tipAngleDeg = Number(spec.tip_angle_deg !== undefined ? spec.tip_angle_deg : 180.0);
    const pitchMm = Math.max(0.75, Math.min(3.0, Number(spec.pitch_mm || 1.25)));
    const matId = toolMaterialOverride || spec.tool_material_id || "carbide_coated";
    const coat = getCoatingColors(matId, etype, flutes);

    const pos = [];
    const nrm = [];
    const col = [];
    const slices = 36;

    function pushTri(p0, p1, p2, n0, n1, n2, c0, c1, c2) {
      const ca = c0, cb = c1 || c0, cc = c2 || c0;
      pos.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      nrm.push(n0[0], n0[1], n0[2], n1[0], n1[1], n1[2], n2[0], n2[1], n2[2]);
      col.push(ca[0], ca[1], ca[2], cb[0], cb[1], cb[2], cc[0], cc[1], cc[2]);
    }

    function addLatheFrustum(z0, z1, r0, r1, color) {
      if (z1 <= z0 + 1e-5) return;
      const dz = z1 - z0;
      const dr = r1 - r0;
      const slopeLen = Math.hypot(dz, dr) || 1.0;
      const nrScale = dz / slopeLen;
      const nzComp = -dr / slopeLen;
      for (let i = 0; i < slices; i++) {
        const a0 = (i / slices) * Math.PI * 2.0;
        const a1 = ((i + 1) / slices) * Math.PI * 2.0;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const p00 = [r0 * c0, r0 * s0, z0];
        const p10 = [r0 * c1, r0 * s1, z0];
        const p11 = [r1 * c1, r1 * s1, z1];
        const p01 = [r1 * c0, r1 * s0, z1];
        const n0 = [c0 * nrScale, s0 * nrScale, nzComp];
        const n1 = [c1 * nrScale, s1 * nrScale, nzComp];
        pushTri(p00, p10, p11, n0, n1, n1, color);
        pushTri(p00, p11, p01, n0, n1, n0, color);
      }
    }

    function addDiskCap(z, rInner, rOuter, upZ, color) {
      if (rOuter <= rInner + 1e-5) return;
      const n = [0, 0, upZ >= 0 ? 1 : -1];
      for (let i = 0; i < slices; i++) {
        const a0 = (i / slices) * Math.PI * 2.0;
        const a1 = ((i + 1) / slices) * Math.PI * 2.0;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        if (rInner <= 1e-5) {
          if (upZ < 0) {
            pushTri([0, 0, z], [rOuter * c1, rOuter * s1, z], [rOuter * c0, rOuter * s0, z], n, n, n, color);
          } else {
            pushTri([0, 0, z], [rOuter * c0, rOuter * s0, z], [rOuter * c1, rOuter * s1, z], n, n, n, color);
          }
        } else {
          const p00 = [rInner * c0, rInner * s0, z];
          const p10 = [rInner * c1, rInner * s1, z];
          const p11 = [rOuter * c1, rOuter * s1, z];
          const p01 = [rOuter * c0, rOuter * s0, z];
          if (upZ < 0) {
            pushTri(p00, p11, p10, n, n, n, color);
            pushTri(p00, p01, p11, n, n, n, color);
          } else {
            pushTri(p00, p10, p11, n, n, n, color);
            pushTri(p00, p11, p01, n, n, n, color);
          }
        }
      }
    }

    function addHelicalFlutedSection(z0, z1, rBase, gulletDepthFrac) {
      if (z1 <= z0 + 1e-4) return;
      const rings = Math.max(4, Math.min(16, Math.ceil((z1 - z0) / Math.max(1.2, 0.35 * rBase))));
      const tanHelix = Math.tan(helixRad);
      function sampleRing(z) {
        const pts = [];
        const nrms = [];
        const cols = [];
        const twist = (z * tanHelix) / Math.max(rBase, 0.4);
        for (let i = 0; i <= slices; i++) {
          const a = (i / slices) * Math.PI * 2.0;
          const wave = 0.5 + 0.5 * Math.sin(flutes * (a - twist));
          const groove = Math.pow(wave, 1.8);
          const rad = rBase * (1.0 - gulletDepthFrac * groove);
          const c = Math.cos(a), s = Math.sin(a);
          pts.push([rad * c, rad * s, z]);
          nrms.push([c, s, 0.0]);
          cols.push([
            coat.land[0] * (1.0 - groove) + coat.gullet[0] * groove,
            coat.land[1] * (1.0 - groove) + coat.gullet[1] * groove,
            coat.land[2] * (1.0 - groove) + coat.gullet[2] * groove,
          ]);
        }
        return { pts, nrms, cols };
      }
      let prev = sampleRing(z0);
      for (let k = 1; k <= rings; k++) {
        const z = z0 + ((z1 - z0) * k) / rings;
        const cur = sampleRing(z);
        for (let i = 0; i < slices; i++) {
          pushTri(prev.pts[i], prev.pts[i + 1], cur.pts[i + 1], prev.nrms[i], prev.nrms[i + 1], cur.nrms[i + 1], prev.cols[i], prev.cols[i + 1], cur.cols[i + 1]);
          pushTri(prev.pts[i], cur.pts[i + 1], cur.pts[i], prev.nrms[i], cur.nrms[i + 1], cur.nrms[i], prev.cols[i], cur.cols[i + 1], cur.cols[i]);
        }
        prev = cur;
      }
    }

    // --- 1. Cutting Tip & Flute Body by StandardToolLibrary Entity Type ---
    if (etype === "probe") {
      // Ruby sphere stylus centered at (0, 0, rCut) + ceramic stem + optical probe body
      const rubyCol = [0.95, 0.14, 0.28];
      const latRings = 12;
      for (let j = 0; j < latRings; j++) {
        const p0 = (j / latRings) * Math.PI;
        const p1 = ((j + 1) / latRings) * Math.PI;
        const z0 = rCut - rCut * Math.cos(p0);
        const z1 = rCut - rCut * Math.cos(p1);
        const rr0 = rCut * Math.sin(p0);
        const rr1 = rCut * Math.sin(p1);
        addLatheFrustum(z0, z1, rr0, rr1, rubyCol);
      }
      const stemR = Math.max(0.45, 0.38 * rCut);
      addLatheFrustum(1.85 * rCut, holderBot * 0.78, stemR, stemR, [0.92, 0.94, 0.96]); // white ceramic stem
      addLatheFrustum(holderBot * 0.78, holderBot, stemR * 1.6, rShank * 1.3, [0.68, 0.73, 0.79]);
      // Optical probe transceiver body + green ready ring
      addLatheFrustum(holderBot, holderBot + 3.0, rHolder * 0.92, rHolder * 0.92, [0.16, 0.88, 0.44]);
    } else if (etype === "face mill" || etype === "slot mill") {
      // Wide shell mill body + indexable golden carbide inserts around perimeter
      const bodyCol = [0.30, 0.35, 0.42];
      const insertCol = [0.98, 0.80, 0.20];
      const bodyTopZ = Math.max(fluteH, 8.0);
      addDiskCap(0.15, 0.0, rCut * 0.88, -1, bodyCol);
      addLatheFrustum(0.15, bodyTopZ, rCut * 0.90, rCut * 0.96, bodyCol);
      addDiskCap(bodyTopZ, rShank, rCut * 0.96, 1, bodyCol);
      // Protruding carbide inserts
      const insH = Math.min(fluteH, 5.2);
      const insSpan = (Math.PI * 2.0) / Math.max(3, flutes) * 0.34;
      for (let f = 0; f < flutes; f++) {
        const aMid = (f / flutes) * Math.PI * 2.0;
        const a0 = aMid - insSpan * 0.5;
        const a1 = aMid + insSpan * 0.5;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const rIn = rCut * 0.78;
        const rOut = rCut * 1.03;
        const b00 = [rOut * c0, rOut * s0, 0.0];
        const b10 = [rOut * c1, rOut * s1, 0.0];
        const b11 = [rOut * c1, rOut * s1, insH];
        const b01 = [rOut * c0, rOut * s0, insH];
        const nOut = [Math.cos(aMid), Math.sin(aMid), 0.0];
        pushTri(b00, b10, b11, nOut, nOut, nOut, insertCol);
        pushTri(b00, b11, b01, nOut, nOut, nOut, insertCol);
        // Bottom face of insert
        const i00 = [rIn * c0, rIn * s0, 0.0];
        const i10 = [rIn * c1, rIn * s1, 0.0];
        pushTri(i00, b10, b00, [0, 0, -1], [0, 0, -1], [0, 0, -1], insertCol);
        pushTri(i00, i10, b10, [0, 0, -1], [0, 0, -1], [0, 0, -1], insertCol);
      }
      if (holderBot > bodyTopZ + 0.2) {
        addLatheFrustum(bodyTopZ, holderBot, rShank, rShank, [0.64, 0.70, 0.77]);
      }
    } else if (etype === "ball end mill" || etype === "lollipop mill") {
      // Hemispherical (or full spherical lollipop) ball nose + helical flutes
      const latMax = etype === "lollipop mill" ? Math.PI * 0.82 : Math.PI * 0.5;
      const latSteps = 9;
      for (let j = 0; j < latSteps; j++) {
        const p0 = (j / latSteps) * latMax;
        const p1 = ((j + 1) / latSteps) * latMax;
        const z0 = rCut * (1.0 - Math.cos(p0));
        const z1 = rCut * (1.0 - Math.cos(p1));
        const rr0 = rCut * Math.sin(p0);
        const rr1 = rCut * Math.sin(p1);
        addLatheFrustum(z0, z1, rr0, rr1, j % 2 === 0 ? coat.land : coat.gullet);
      }
      const zSphereEnd = rCut * (1.0 - Math.cos(latMax));
      if (etype === "lollipop mill") {
        const rNeck = Math.min(rShank, rCut * 0.55);
        addLatheFrustum(zSphereEnd, holderBot, rNeck, rShank, [0.72, 0.77, 0.83]);
      } else {
        addHelicalFlutedSection(rCut, fluteH, rCut, 0.14);
        addLatheFrustum(fluteH, holderBot, rCut * 0.96, rShank, [0.70, 0.75, 0.82]);
      }
    } else if (etype === "thread mill" || etype === "tap right hand") {
      // 60-degree V-thread annular cutting teeth spaced by pitchMm
      const cutTop = etype === "thread mill" ? Math.min(fluteH, Math.max(3.0 * pitchMm, 0.55 * fluteH)) : fluteH;
      const nTeeth = Math.max(3, Math.min(18, Math.round(cutTop / pitchMm)));
      const stepZ = cutTop / nTeeth;
      const rRoot = rCut * 0.68;
      addDiskCap(0.0, 0.0, rRoot, -1, coat.land);
      for (let t = 0; t < nTeeth; t++) {
        const zA = t * stepZ;
        const zMid = zA + 0.5 * stepZ;
        const zB = zA + stepZ;
        addLatheFrustum(zA, zMid, rRoot, rCut, coat.land);
        addLatheFrustum(zMid, zB, rCut, rRoot, coat.gullet);
      }
      if (cutTop < fluteH - 0.1) {
        // Relieved thread-mill neck allowing radial clearance inside deep threads
        addLatheFrustum(cutTop, fluteH, rCut * 0.64, rCut * 0.64, coat.gullet);
      }
      addLatheFrustum(fluteH, holderBot, rCut * 0.64, rShank, [0.70, 0.75, 0.82]);
    } else if (etype === "dovetail mill") {
      // Inverted dovetail frustum (wider at bottom z=0, necked at z=fluteH)
      const rNeck = Math.min(rShank, rCut * 0.52);
      addDiskCap(0.0, 0.0, rCut, -1, coat.land);
      addLatheFrustum(0.0, fluteH, rCut, rNeck, coat.land);
      addLatheFrustum(fluteH, holderBot, rNeck, rShank, [0.70, 0.75, 0.82]);
    } else if (
      etype === "drill" ||
      etype === "spot drill" ||
      etype === "center drill" ||
      etype === "chamfer mill" ||
      etype === "counter sink" ||
      (tipAngleDeg > 15.0 && tipAngleDeg < 172.0)
    ) {
      // Conical point (90° spot/chamfer, 82° countersink, 118°/135° twist drill)
      const effDeg = Math.max(30.0, Math.min(160.0, tipAngleDeg < 172.0 ? tipAngleDeg : (etype === "drill" ? 135.0 : 90.0)));
      const halfRad = (effDeg * Math.PI) / 360.0;
      const zCone = Math.min(fluteH * 0.92, rCut / Math.max(0.18, Math.tan(halfRad)));
      // Fluted conical point
      const coneRings = 5;
      for (let j = 0; j < coneRings; j++) {
        const f0 = j / coneRings;
        const f1 = (j + 1) / coneRings;
        addLatheFrustum(f0 * zCone, f1 * zCone, f0 * rCut, f1 * rCut, j % 2 === 0 ? coat.land : coat.gullet);
      }
      if (etype === "drill") {
        addHelicalFlutedSection(zCone, fluteH, rCut, 0.24);
      } else {
        addHelicalFlutedSection(zCone, fluteH, rCut, 0.12);
      }
      addLatheFrustum(fluteH, holderBot, rCut, rShank, [0.70, 0.75, 0.82]);
    } else if (rc > 0.06) {
      // Bull-nose / corner-radius end mill: flat core + quarter-torus corner + helical flutes
      const rFlat = Math.max(0.0, rCut - rc);
      addDiskCap(0.0, 0.0, rFlat, -1, coat.land);
      const torSteps = 5;
      for (let j = 0; j < torSteps; j++) {
        const p0 = (j / torSteps) * (Math.PI * 0.5);
        const p1 = ((j + 1) / torSteps) * (Math.PI * 0.5);
        const z0 = rc * (1.0 - Math.cos(p0));
        const z1 = rc * (1.0 - Math.cos(p1));
        const rr0 = rFlat + rc * Math.sin(p0);
        const rr1 = rFlat + rc * Math.sin(p1);
        addLatheFrustum(z0, z1, rr0, rr1, coat.land);
      }
      addHelicalFlutedSection(rc, fluteH, rCut, 0.15);
      addLatheFrustum(fluteH, holderBot, rCut, rShank, [0.70, 0.75, 0.82]);
    } else {
      // Flat end mill / reamer / counterbore: crisp flat face + helical flutes + ground shank
      addDiskCap(0.0, 0.0, rCut, -1, coat.land);
      addHelicalFlutedSection(0.0, fluteH, rCut, etype === "reamer" ? 0.08 : 0.16);
      addLatheFrustum(fluteH, holderBot, rCut, rShank, [0.70, 0.75, 0.82]);
    }

    // --- 2. Stepped ER Collet Nut + V-Flange BT30 / CAT40 Spindle Tool Holder ---
    const nutCol = [0.16, 0.18, 0.22];
    const bodyCol = [0.27, 0.31, 0.38];
    const flangeCol = [0.36, 0.41, 0.49];
    const rNut = Math.max(rShank + 2.2, rHolder * 0.62);
    const zNutTop = holderBot + holderH * 0.24;
    const zNoseTop = holderBot + holderH * 0.58;
    const zGrooveBot = holderBot + holderH * 0.68;
    const zGrooveTop = holderBot + holderH * 0.80;
    const zTop = holderBot + holderH;

    addDiskCap(holderBot, Math.min(rShank, rNut * 0.85), rNut * 0.88, -1, nutCol);
    addLatheFrustum(holderBot, zNutTop, rNut * 0.88, rNut, nutCol);
    addLatheFrustum(zNutTop, zNoseTop, rNut, rHolder * 0.90, bodyCol);
    addLatheFrustum(zNoseTop, zGrooveBot, rHolder, rHolder, flangeCol);
    // ATC V-flange gripper groove
    addLatheFrustum(zGrooveBot, 0.5 * (zGrooveBot + zGrooveTop), rHolder, rHolder * 0.82, nutCol);
    addLatheFrustum(0.5 * (zGrooveBot + zGrooveTop), zGrooveTop, rHolder * 0.82, rHolder, nutCol);
    addLatheFrustum(zGrooveTop, zTop, rHolder, rHolder * 0.94, flangeCol);
    addDiskCap(zTop, 0.0, rHolder * 0.94, 1, bodyCol);

    return {
      positions: new Float32Array(pos),
      normals: new Float32Array(nrm),
      colors: new Float32Array(col),
      vertexCount: pos.length / 3,
    };
  }

  // --- Tight Auto-Fit Trajectory Color Ramps & 3D Thick Tube Extrusion ---
  const FIVE_STOP_RAMP = [
    [0.145, 0.388, 0.922], // #2563eb Cobalt Blue (min)
    [0.024, 0.714, 0.831], // #06b6d4 Bright Cyan (25%)
    [0.133, 0.773, 0.369], // #22c55e Emerald Green (50%)
    [0.980, 0.800, 0.082], // #facc15 Vivid Gold (75%)
    [0.937, 0.267, 0.267], // #ef4444 Crimson Red (max)
  ];

  function sampleFiveStopRgb(tNorm) {
    const t = Math.max(0.0, Math.min(1.0, tNorm));
    const scaled = t * 4.0;
    const idx = Math.min(3, Math.floor(scaled));
    const u = scaled - idx;
    const c0 = FIVE_STOP_RAMP[idx];
    const c1 = FIVE_STOP_RAMP[idx + 1];
    return [
      c0[0] + (c1[0] - c0[0]) * u,
      c0[1] + (c1[1] - c0[1]) * u,
      c0[2] + (c1[2] - c0[2]) * u,
    ];
  }

  function extractSegmentMetric(seg, mode) {
    if (mode === "chipload") {
      return seg.engaged ? Number(seg.hex_mm || 0.0) : null;
    }
    if (mode === "engagement") {
      return seg.engaged ? Number(seg.ae_pct || 0.0) : null;
    }
    if (mode === "mrr") {
      return seg.engaged ? Number(seg.mrr_mm3_s || 0.0) : null;
    }
    if (mode === "feedrate") {
      return Number(seg.feed_mm_s || 0.0);
    }
    return null;
  }

  function computeTightColorRange(plans, mode) {
    if (mode === "engaged_air" || mode === "tool_number" || mode === "operation") {
      return { min: 1.0, max: 4.0, span: 3.0, constant: false };
    }
    let vMin = Infinity;
    let vMax = -Infinity;
    for (let p = 0; p < plans.length; p++) {
      const segs = (plans[p] && plans[p].segments) || [];
      for (let i = 0; i < segs.length; i++) {
        const s = segs[i];
        if (s.kind === "rapid") continue;
        const val = extractSegmentMetric(s, mode);
        if (val !== null && Number.isFinite(val)) {
          if (val < vMin) vMin = val;
          if (val > vMax) vMax = val;
        }
      }
    }
    if (!Number.isFinite(vMin) || !Number.isFinite(vMax)) {
      return { min: 0.0, max: 1.0, span: 1.0, constant: true };
    }
    const rawSpan = vMax - vMin;
    const relTol = Math.max(1e-5, Math.abs(vMax) * 1e-4);
    if (rawSpan <= relTol) {
      return { min: vMin, max: vMax, span: 1.0, constant: true };
    }
    return { min: vMin, max: vMax, span: rawSpan, constant: false };
  }

  const TOOL_PALETTE_RGB = [
    [0.145, 0.520, 0.960], // T1: Cobalt Blue
    [0.133, 0.800, 0.420], // T2: Emerald Green
    [0.980, 0.780, 0.120], // T3: Vivid Gold
    [0.940, 0.340, 0.560], // T4: Crimson Rose
    [0.060, 0.760, 0.860], // T5: Bright Cyan
    [0.680, 0.420, 0.960], // T6: Violet Orchid
    [0.980, 0.520, 0.160], // T7: Tangerine Orange
    [0.520, 0.880, 0.240], // T8: Lime Chartreuse
  ];

  const OP_PALETTE_RGB = [
    [0.220, 0.741, 0.973], // Op 1: #38bdf8 Sky Cyan
    [0.204, 0.827, 0.600], // Op 2: #34d399 Emerald
    [0.984, 0.749, 0.141], // Op 3: #fbbf24 Amber Gold
    [0.957, 0.447, 0.714], // Op 4: #f472b6 Flamingo Rose
    [0.655, 0.545, 0.980], // Op 5: #a78bfa Violet
    [0.984, 0.573, 0.235], // Op 6: #fb923c Tangerine
    [0.176, 0.831, 0.749], // Op 7: #2dd4bf Teal
    [0.973, 0.443, 0.443], // Op 8: #f87171 Coral Red
    [0.639, 0.902, 0.208], // Op 9: #a3e635 Lime
    [0.376, 0.647, 0.980], // Op 10: #60a5fa Periwinkle
    [0.910, 0.475, 0.976], // Op 11: #e879f9 Fuchsia
    [0.992, 0.878, 0.278], // Op 12: #fde047 Lemon
  ];

  const OP_PALETTE_HEX = [
    "#38bdf8",
    "#34d399",
    "#fbbf24",
    "#f472b6",
    "#a78bfa",
    "#fb923c",
    "#2dd4bf",
    "#f87171",
    "#a3e635",
    "#60a5fa",
    "#e879f9",
    "#fde047",
  ];

  function cutSegmentColorRgb(seg, mode, rangeObj, opColorIndexMap) {
    if (mode === "engaged_air") {
      return seg.engaged ? [0.18, 0.85, 0.45] : [0.94, 0.39, 0.58];
    }
    if (mode === "tool_number") {
      const tNum = Number(seg.tool_number || 1);
      if (tNum >= 90 || String(seg.entity_type || "").toLowerCase() === "probe") {
        return [0.95, 0.18, 0.24]; // Spindle Probe T99
      }
      const tIdx = Math.max(0, (tNum - 1) % TOOL_PALETTE_RGB.length);
      return TOOL_PALETTE_RGB[tIdx];
    }
    if (mode === "operation") {
      const opKey = resolveSegmentOpKey(seg);
      let cIdx = 0;
      if (opColorIndexMap && opColorIndexMap.has(opKey)) {
        cIdx = opColorIndexMap.get(opKey);
      } else if (seg.op_index !== undefined) {
        cIdx = Math.max(0, Number(seg.op_index) - 1);
      } else {
        cIdx = Math.max(0, Number(seg.tool_number || 1) - 1);
      }
      return OP_PALETTE_RGB[cIdx % OP_PALETTE_RGB.length];
    }
    const val = extractSegmentMetric(seg, mode);
    if (val === null) {
      // Unengaged G1 air-cut move in CWE modes: distinct cool slate-indigo so it doesn't skew tight scale
      return [0.44, 0.50, 0.64];
    }
    if (rangeObj.constant) {
      return FIVE_STOP_RAMP[2]; // Emerald mid-point when trajectory value is uniform (e.g. Same-Speed feedrate)
    }
    return sampleFiveStopRgb((val - rangeObj.min) / rangeObj.span);
  }

  // --- Multi-Operation Keying & Kurt DX6 Workholding Geometry Synthesis ---
  function resolveSegmentOpKey(seg) {
    if (!seg) return "op_default";
    if (seg.op_id && String(seg.op_id).trim() !== "") {
      return String(seg.op_id);
    }
    const sIdx = Number(seg.setup_index || 1);
    const tNum = Number(seg.tool_number || 1);
    return "s" + sIdx + "_t" + tNum;
  }

  function formatShortOpKind(rawKind, rawId) {
    const k = String(rawKind || rawId || "mill").toLowerCase();
    const map = {
      facing: "Face",
      pocket_adaptive: "Rough",
      pocket_rest: "Rest",
      levelset_rough: "Rough",
      levelset_rest_finish: "Rest Finish",
      helical_bore: "Helix Bore",
      exterior_contour: "Contour",
      freeform_3d_surfacing: "3D Surf",
      spot_drill: "Spot",
      peck_drill: "Drill",
      counterbore_mill: "C-Bore",
      countersink: "C-Sink",
      chamfer_contour: "Chamfer",
      thread_mill: "Thread",
      op2_hat_facing: "Op2 Hat Face",
      op2_adaptive_pocket: "Op2 Pocket",
      wips_probe: "WIPS Probe",
    };
    if (map[k]) return map[k];
    const clean = k.replace(/^s[12]_op\d+_/, "").replace(/_/g, " ").trim();
    return clean.length > 14 ? clean.slice(0, 14) : clean || "Mill";
  }

  function deriveCaseOperations(caseData, variant) {
    const segs = (variant && variant.plan && variant.plan.segments) || [];
    const opsTable = (caseData && Array.isArray(caseData.operations_table) && caseData.operations_table) || [];
    const tableByOpId = new Map();
    for (let i = 0; i < opsTable.length; i++) {
      const row = opsTable[i];
      if (row && row.op_id) {
        tableByOpId.set(String(row.op_id), row);
      }
    }

    const opMap = new Map();
    const ordered = [];
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      const key = resolveSegmentOpKey(s);
      let entry = opMap.get(key);
      if (!entry) {
        const meta = tableByOpId.get(key) || null;
        const setupIndex = Number((meta && meta.setup_index) || s.setup_index || 1);
        const workOffset = String(
          (meta && meta.work_offset) || s.work_offset || (setupIndex === 2 ? "G55" : "G54")
        );
        const toolNumber = Number((meta && meta.tool_number) || s.tool_number || (caseData && caseData.tool && caseData.tool.tool_number) || 1);
        const toolSpec = resolveToolSpecForSegment(caseData, s);
        const opIndex = ordered.length + 1;
        const cIdx = (opIndex - 1) % OP_PALETTE_RGB.length;
        const rawKind = (meta && meta.op_kind) || s.op_kind || (caseData && caseData.family) || "milling";
        entry = {
          key,
          opIndex,
          opId: key,
          opKind: String(rawKind),
          shortKind: formatShortOpKind(rawKind, key),
          setupIndex,
          workOffset,
          toolNumber,
          toolLabel: (meta && meta.tool_label) || formatActiveToolLabel(toolSpec),
          notes: (meta && meta.notes) || "",
          t0S: Number(s.t0_s !== undefined ? s.t0_s : s.t0 || 0.0),
          t1S: Number(s.t1_s !== undefined ? s.t1_s : s.t1 || 0.0),
          firstCutT0S: null,
          durationS: 0.0,
          firstSegIdx: i,
          lastSegIdx: i,
          segCount: 0,
          cutSegCount: 0,
          colorIdx: cIdx,
          colorRgb: OP_PALETTE_RGB[cIdx],
          colorHex: OP_PALETTE_HEX[cIdx],
        };
        opMap.set(key, entry);
        ordered.push(entry);
      }
      const st0 = Number(s.t0_s !== undefined ? s.t0_s : s.t0 || 0.0);
      const st1 = Number(s.t1_s !== undefined ? s.t1_s : s.t1 || 0.0);
      if (st0 < entry.t0S) entry.t0S = st0;
      if (st1 > entry.t1S) entry.t1S = st1;
      entry.lastSegIdx = i;
      entry.segCount += 1;
      if (s.kind !== "rapid") {
        entry.cutSegCount += 1;
        if (entry.firstCutT0S === null) {
          entry.firstCutT0S = st0;
        }
      }
    }

    for (let i = 0; i < ordered.length; i++) {
      ordered[i].durationS = Math.max(0.0, ordered[i].t1S - ordered[i].t0S);
      if (ordered[i].firstCutT0S === null) {
        ordered[i].firstCutT0S = ordered[i].t0S;
      }
    }
    return ordered;
  }

  // Resolve TAMP Kurt DX6 workholding specification (uses exported caseData.workholding or synthesizes exact DX6 dimensions).
  function resolveCaseWorkholding(caseData) {
    const sMin = (caseData && caseData.stock_min_mm) || [-25, -20, -12];
    const sMax = (caseData && caseData.stock_max_mm) || [25, 20, 0];
    const sx0 = Number(sMin[0]), sy0 = Number(sMin[1]), sz0 = Number(sMin[2]);
    const sx1 = Number(sMax[0]), sy1 = Number(sMax[1]), sz1 = Number(sMax[2]);
    const lx = Math.max(1.0, sx1 - sx0);
    const ly = Math.max(1.0, sy1 - sy0);
    const lz = Math.max(2.0, sz1 - sz0);

    if (caseData && caseData.workholding && caseData.workholding.setups) {
      return caseData.workholding;
    }

    const jawWidth = 152.4;      // 6.000"
    const jawHeight = 44.45;     // 1.750"
    const jawThickness = 19.05;  // 0.750"
    const softJawThick = 25.4;   // 1.000"
    const maxOpenStd = 228.6;    // 9.000"
    const bedHeight = 73.025;    // 2.875"
    const bodyLength = 429.26;   // 16.90"
    const parallelThick = 3.175; // 0.125"

    const pMin = (caseData && caseData.pocket_min_mm) || [sx0 + 2, sy0 + 2, sz0 + 2.5];
    const pMax = (caseData && caseData.pocket_max_mm) || [sx1 - 2, sy1 - 2, sz1];
    const cadMeta = (caseData && caseData.cad_metadata) || {};
    const partDims = Array.isArray(cadMeta.part_dims_mm) ? cadMeta.part_dims_mm : [lx - 2.0, ly - 2.0, Math.max(2.0, lz - 3.5)];
    const hasOp2 = Boolean(
      cadMeta.has_op2_flip ||
        ((caseData && caseData.operations_table) || []).some((op) => Number(op.setup_index || 1) >= 2)
    );

    const floorMargin = Number(pMin[2]) - sz0;
    const gripZ = Math.max(1.5, Math.min(3.5, floorMargin > 1.8 ? Math.min(2.5, floorMargin * 0.55) : 2.0));
    const clampAxis = ly > maxOpenStd && lx <= maxOpenStd ? "X" : "Y";
    const op1SpanMin = clampAxis === "Y" ? sy0 : sx0;
    const op1SpanMax = clampAxis === "Y" ? sy1 : sx1;
    const crossMin = clampAxis === "Y" ? sx0 : sy0;
    const crossMax = clampAxis === "Y" ? sx1 : sy1;
    const jawOpenOp1 = op1SpanMax - op1SpanMin;
    const jawMounting = jawOpenOp1 > maxOpenStd ? "reversed_exterior" : "standard_interior";

    const jawTopOp1 = sz0 + gripZ;
    const bedZOp1 = jawTopOp1 - jawHeight;
    const tableZOp1 = bedZOp1 - bedHeight;
    const parallelHeight = Math.max(6.0, jawHeight - gripZ);

    const op2HalfSpan = 0.5 * Math.max(6.0, Number(clampAxis === "Y" ? partDims[1] : partDims[0]) || (op1SpanMax - op1SpanMin));
    const op2HalfCross = 0.5 * Math.max(6.0, Number(clampAxis === "Y" ? partDims[0] : partDims[1]) || (crossMax - crossMin));
    const jawTopOp2 = Math.min(sz1 - gripZ - 1.2, sz1 - 1.5);
    const seatZOp2 = Math.min(sz0 + Math.max(0.0, sz1 - Number(pMax[2] || 0.0)), jawTopOp2 - 2.5);
    const bedZOp2 = jawTopOp2 - jawHeight;
    const tableZOp2 = bedZOp2 - bedHeight;

    return {
      vise_model: "kurt_dx6_6in",
      vise_label: 'Kurt DX6 CrossOver 6" Machine Vise',
      jaw_width_mm: jawWidth,
      jaw_height_mm: jawHeight,
      jaw_thickness_mm: jawThickness,
      soft_jaw_thickness_mm: softJawThick,
      max_opening_standard_mm: maxOpenStd,
      max_opening_reversed_mm: 386.0,
      bed_height_mm: bedHeight,
      body_length_mm: bodyLength,
      clamping_axis: clampAxis,
      jaw_mounting: jawMounting,
      engagement_width_mm: Math.min(jawWidth, crossMax - crossMin),
      has_op2_soft_jaws: hasOp2,
      summary:
        'Kurt DX6 6" (' +
        clampAxis +
        "-clamp, " +
        jawOpenOp1.toFixed(1) +
        " mm open, " +
        gripZ.toFixed(1) +
        " mm bite on " +
        parallelHeight.toFixed(2) +
        " mm parallels" +
        (hasOp2 ? " → G55 6061 Soft Jaws)" : ")"),
      setups: {
        "1": {
          setup_index: 1,
          work_offset: "G54",
          jaw_type: "hardened_steel_parallels",
          label: "Kurt DX6 · G54 Hard Jaws + " + parallelHeight.toFixed(2) + " mm Parallels",
          short_label: "Kurt DX6 · G54 Hard Jaws",
          clamping_axis: clampAxis,
          jaw_mounting: jawMounting,
          jaw_opening_mm: jawOpenOp1,
          clamped_span_min_mm: op1SpanMin,
          clamped_span_max_mm: op1SpanMax,
          cross_span_min_mm: crossMin,
          cross_span_max_mm: crossMax,
          grip_z_mm: gripZ,
          jaw_top_z_mm: jawTopOp1,
          parallel_height_mm: parallelHeight,
          parallel_thickness_mm: parallelThick,
          seat_z_mm: sz0,
          bed_z_mm: bedZOp1,
          table_z_mm: tableZOp1,
          jaw_thickness_mm: jawThickness,
          jaw_height_mm: jawHeight,
        },
        "2": {
          setup_index: 2,
          work_offset: "G55",
          jaw_type: "machined_6061_soft_jaws",
          label: "Kurt DX6 · G55 Contoured 6061 Soft Jaws",
          short_label: "Kurt DX6 · G55 Soft Jaws",
          clamping_axis: clampAxis,
          jaw_mounting: jawMounting,
          jaw_opening_mm: 2.0 * op2HalfSpan,
          clamped_span_min_mm: -op2HalfSpan,
          clamped_span_max_mm: op2HalfSpan,
          cross_span_min_mm: -op2HalfCross,
          cross_span_max_mm: op2HalfCross,
          grip_z_mm: Math.max(2.0, jawTopOp2 - seatZOp2),
          jaw_top_z_mm: jawTopOp2,
          parallel_height_mm: 0.0,
          parallel_thickness_mm: 0.0,
          seat_z_mm: seatZOp2,
          bed_z_mm: bedZOp2,
          table_z_mm: tableZOp2,
          jaw_thickness_mm: softJawThick,
          jaw_height_mm: jawHeight,
        },
      },
    };
  }

  // Procedural Clean-Room 3D Triangle Mesh Generator for the Kurt DX6 CrossOver 6" CNC Machine Vise.
  // Built in local vise coordinates (U = cross-jaw width [-76.2..+76.2], V = clamping axis [-rear fixed..+front hex], Z = world Z),
  // then mapped cleanly to world (X, Y, Z) for either Y-clamping (default) or 90-deg X-clamping!
  function buildKurtDx6ViseMesh(caseData, setupIndex) {
    const wh = resolveCaseWorkholding(caseData);
    const setupKey = Number(setupIndex || 1) >= 2 ? "2" : "1";
    const st = (wh.setups && wh.setups[setupKey]) || wh.setups["1"];
    const clampAxis = String(st.clamping_axis || wh.clamping_axis || "Y");
    const isSoftJaws = st.jaw_type === "machined_6061_soft_jaws" || setupKey === "2";
    const isReversed = st.jaw_mounting === "reversed_exterior";

    const jawW = Number(wh.jaw_width_mm || 152.4);
    const halfW = 0.5 * jawW; // 76.2 mm
    const jawH = Number(st.jaw_height_mm || wh.jaw_height_mm || 44.45);
    const jawT = Number(st.jaw_thickness_mm || (isSoftJaws ? 25.4 : 19.05));
    const bedH = Number(wh.bed_height_mm || 73.025);
    const spanMin = Number(st.clamped_span_min_mm);
    const spanMax = Number(st.clamped_span_max_mm);
    const crossMin = Number(st.cross_span_min_mm || -25.0);
    const crossMax = Number(st.cross_span_max_mm || 25.0);
    const jawTopZ = Number(st.jaw_top_z_mm);
    const bedZ = Number(st.bed_z_mm !== undefined ? st.bed_z_mm : jawTopZ - jawH);
    const tableZ = Number(st.table_z_mm !== undefined ? st.table_z_mm : bedZ - bedH);
    const seatZ = Number(st.seat_z_mm !== undefined ? st.seat_z_mm : jawTopZ - Number(st.grip_z_mm || 2.0));
    const parH = Number(st.parallel_height_mm || 0.0);
    const parT = Number(st.parallel_thickness_mm || 3.175);

    const posArr = [];
    const nrmArr = [];
    const colArr = [];

    function toWorldPt(u, v, z) {
      // Y-clamp: U -> +X, V -> +Y, Z -> +Z (det = +1)
      // X-clamp: U -> -Y, V -> +X, Z -> +Z (90 deg rotation in XY, det = +1 preserves triangle winding!)
      if (clampAxis === "X") {
        return [v, -u, z];
      }
      return [u, v, z];
    }

    function toWorldVec(nu, nv, nz) {
      if (clampAxis === "X") {
        return [nv, -nu, nz];
      }
      return [nu, nv, nz];
    }

    function pushTriLocal(p0, p1, p2, rgb) {
      const w0 = toWorldPt(p0[0], p0[1], p0[2]);
      const w1 = toWorldPt(p1[0], p1[1], p1[2]);
      const w2 = toWorldPt(p2[0], p2[1], p2[2]);
      const e1x = w1[0] - w0[0], e1y = w1[1] - w0[1], e1z = w1[2] - w0[2];
      const e2x = w2[0] - w0[0], e2y = w2[1] - w0[1], e2z = w2[2] - w0[2];
      let nx = e1y * e2z - e1z * e2y;
      let ny = e1z * e2x - e1x * e2z;
      let nz = e1x * e2y - e1y * e2x;
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen < 1e-9) return;
      nx /= nLen; ny /= nLen; nz /= nLen;
      posArr.push(w0[0], w0[1], w0[2], w1[0], w1[1], w1[2], w2[0], w2[1], w2[2]);
      nrmArr.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
      colArr.push(rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2]);
    }

    function pushQuadLocal(p0, p1, p2, p3, rgb) {
      pushTriLocal(p0, p1, p2, rgb);
      pushTriLocal(p0, p2, p3, rgb);
    }

    function addBoxUVZ(u0, u1, v0, v1, z0, z1, rgb, topRgb) {
      if (u1 <= u0 + 1e-5 || v1 <= v0 + 1e-5 || z1 <= z0 + 1e-5) return;
      const cTop = topRgb || rgb;
      // +Z top
      pushQuadLocal([u0, v0, z1], [u1, v0, z1], [u1, v1, z1], [u0, v1, z1], cTop);
      // -Z bottom
      pushQuadLocal([u0, v1, z0], [u1, v1, z0], [u1, v0, z0], [u0, v0, z0], rgb);
      // -V rear
      pushQuadLocal([u1, v0, z0], [u1, v0, z1], [u0, v0, z1], [u0, v0, z0], rgb);
      // +V front
      pushQuadLocal([u0, v1, z0], [u0, v1, z1], [u1, v1, z1], [u1, v1, z0], rgb);
      // -U left
      pushQuadLocal([u0, v0, z0], [u0, v0, z1], [u0, v1, z1], [u0, v1, z0], rgb);
      // +U right
      pushQuadLocal([u1, v1, z0], [u1, v1, z1], [u1, v0, z1], [u1, v0, z0], rgb);
    }

    // Sloped gusset prism on -V or +V side (characteristic Kurt DX6 angled fixed-tower & movable-carriage ribs)
    function addGussetRibUVZ(u0, u1, vBaseStart, vBaseEnd, vTopStart, vTopEnd, z0, z1, rgb) {
      const p00 = [u0, vBaseStart, z0], p10 = [u1, vBaseStart, z0];
      const p11 = [u1, vBaseEnd, z0], p01 = [u0, vBaseEnd, z0];
      const q00 = [u0, vTopStart, z1], q10 = [u1, vTopStart, z1];
      const q11 = [u1, vTopEnd, z1], q01 = [u0, vTopEnd, z1];
      pushQuadLocal(q00, q10, q11, q01, rgb); // top
      pushQuadLocal(p01, p11, p10, p00, rgb); // bottom
      pushQuadLocal(p10, q10, q00, p00, rgb); // -V sloped/vertical face
      pushQuadLocal(p01, q01, q11, p11, rgb); // +V sloped/vertical face
      pushQuadLocal(p00, q00, q01, p01, rgb); // -U side
      pushQuadLocal(p11, q11, q10, p10, rgb); // +U side
    }

    // Cylinder / Hex-prism along V axis (leadscrew, thrust collar, hex drive nut, cap-screw heads)
    function addCylinderAlongV(uCenter, zCenter, v0, v1, radius, nSides, rgb, capEnds) {
      for (let i = 0; i < nSides; i++) {
        const a0 = (2.0 * Math.PI * i) / nSides;
        const a1 = (2.0 * Math.PI * (i + 1)) / nSides;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const uA = uCenter + radius * c0, zA = zCenter + radius * s0;
        const uB = uCenter + radius * c1, zB = zCenter + radius * s1;
        // Outward normal around +V axis requires winding [uA,v0] -> [uA,v1] -> [uB,v1] -> [uB,v0]
        const wA0 = toWorldPt(uA, v0, zA);
        const wA1 = toWorldPt(uA, v1, zA);
        const wB1 = toWorldPt(uB, v1, zB);
        const wB0 = toWorldPt(uB, v0, zB);
        const nMid = toWorldVec(0.5 * (c0 + c1), 0.0, 0.5 * (s0 + s1));
        const nl = Math.hypot(nMid[0], nMid[1], nMid[2]) || 1.0;
        const nx = nMid[0] / nl, ny = nMid[1] / nl, nz = nMid[2] / nl;
        posArr.push(
          wA0[0], wA0[1], wA0[2], wA1[0], wA1[1], wA1[2], wB1[0], wB1[1], wB1[2],
          wA0[0], wA0[1], wA0[2], wB1[0], wB1[1], wB1[2], wB0[0], wB0[1], wB0[2]
        );
        for (let k = 0; k < 6; k++) {
          nrmArr.push(nx, ny, nz);
          colArr.push(rgb[0], rgb[1], rgb[2]);
        }
        if (capEnds) {
          pushTriLocal([uCenter, v1, zCenter], [uB, v1, zB], [uA, v1, zA], rgb);
          pushTriLocal([uCenter, v0, zCenter], [uA, v0, zA], [uB, v0, zB], rgb);
        }
      }
    }

    // Material palette (Kurt DX6 CrossOver)
    const kurtBlue = [0.115, 0.365, 0.665];
    const kurtBlueDark = [0.085, 0.275, 0.520];
    const groundWaySteel = [0.710, 0.750, 0.800];
    const castChannelDark = [0.160, 0.195, 0.245];
    const hardenedJawSteel = [0.245, 0.275, 0.325];
    const hardenedJawFace = [0.330, 0.370, 0.430];
    const softJawAlum = [0.700, 0.770, 0.850];
    const softJawSeat = [0.800, 0.860, 0.925];
    const parallelSteel = [0.820, 0.855, 0.900];
    const screwSteel = [0.600, 0.645, 0.700];
    const blackOxideBolt = [0.120, 0.135, 0.160];

    // Fixed & movable jaw plate positions along V:
    // Standard interior mounting: jaw plates sit between fixed tower (-V) and movable carriage (+V).
    // Reversed exterior mounting: jaw plates bolt onto the outer faces for extra-wide stock (> 228.6 mm).
    let fixedPlateV0, fixedPlateV1, fixedTowerFrontV, fixedTowerRearV;
    let movPlateV0, movPlateV1, movCarriageFrontV, movCarriageRearV;

    if (!isReversed) {
      fixedPlateV1 = spanMin;
      fixedPlateV0 = spanMin - jawT;
      fixedTowerFrontV = fixedPlateV0;
      fixedTowerRearV = fixedTowerFrontV - 66.0;

      movPlateV0 = spanMax;
      movPlateV1 = spanMax + jawT;
      movCarriageRearV = movPlateV1;
      movCarriageFrontV = movCarriageRearV + 68.0;
    } else {
      fixedPlateV1 = spanMin;
      fixedPlateV0 = spanMin - jawT;
      fixedTowerRearV = fixedPlateV1;
      fixedTowerFrontV = fixedTowerRearV + 62.0;

      movPlateV0 = spanMax;
      movPlateV1 = spanMax + jawT;
      movCarriageFrontV = movPlateV0;
      movCarriageRearV = movCarriageFrontV - 64.0;
    }

    const bodyRearV = Math.min(fixedTowerRearV - 10.0, fixedPlateV0 - 12.0);
    const minBodyLen = Number(wh.body_length_mm || 429.26);
    const bodyFrontV = Math.max(bodyRearV + minBodyLen, movCarriageFrontV + 34.0);

    // 1. Kurt DX6 Ductile-Iron Base Frame + Precision Ground Side Ways + Central Chip/Screw Channel
    const wayRailW = 41.0;
    const channelHalfU = halfW - wayRailW; // 35.2 mm central screw slot
    const channelFloorZ = tableZ + 26.0;
    const wayPlateThick = 5.5;

    // Lower solid base floor & side walls
    addBoxUVZ(-halfW, halfW, bodyRearV, bodyFrontV, tableZ, channelFloorZ, kurtBlueDark, castChannelDark);
    addBoxUVZ(-halfW, -channelHalfU, bodyRearV, bodyFrontV, channelFloorZ, bedZ - wayPlateThick, kurtBlue, kurtBlue);
    addBoxUVZ(channelHalfU, halfW, bodyRearV, bodyFrontV, channelFloorZ, bedZ - wayPlateThick, kurtBlue, kurtBlue);
    // Precision-ground hardened bed ways on top of left & right rails
    addBoxUVZ(-halfW, -channelHalfU, bodyRearV + 6.0, bodyFrontV - 8.0, bedZ - wayPlateThick, bedZ, groundWaySteel, groundWaySteel);
    addBoxUVZ(channelHalfU, halfW, bodyRearV + 6.0, bodyFrontV - 8.0, bedZ - wayPlateThick, bedZ, groundWaySteel, groundWaySteel);
    // Front & rear bed end bridges
    addBoxUVZ(-channelHalfU, channelHalfU, bodyRearV, bodyRearV + 26.0, channelFloorZ, bedZ, kurtBlue, groundWaySteel);
    addBoxUVZ(-channelHalfU, channelHalfU, bodyFrontV - 24.0, bodyFrontV, channelFloorZ, bedZ - 2.0, kurtBlue, kurtBlue);

    // Hold-down toe-clamp side ears / bolt flanges along tableZ (signature CNC vise table mounting slots)
    const flangeH = 15.5;
    const flangeOut = 17.5;
    const midV = 0.5 * (bodyRearV + bodyFrontV);
    for (const vCenter of [bodyRearV + 58.0, midV, bodyFrontV - 62.0]) {
      addBoxUVZ(-halfW - flangeOut, -halfW, vCenter - 22.0, vCenter - 6.0, tableZ, tableZ + flangeH, kurtBlueDark, groundWaySteel);
      addBoxUVZ(-halfW - flangeOut, -halfW, vCenter + 6.0, vCenter + 22.0, tableZ, tableZ + flangeH, kurtBlueDark, groundWaySteel);
      addBoxUVZ(-halfW - flangeOut, -halfW, vCenter - 6.0, vCenter + 6.0, tableZ, tableZ + 6.5, castChannelDark, castChannelDark);

      addBoxUVZ(halfW, halfW + flangeOut, vCenter - 22.0, vCenter - 6.0, tableZ, tableZ + flangeH, kurtBlueDark, groundWaySteel);
      addBoxUVZ(halfW, halfW + flangeOut, vCenter + 6.0, vCenter + 22.0, tableZ, tableZ + flangeH, kurtBlueDark, groundWaySteel);
      addBoxUVZ(halfW, halfW + flangeOut, vCenter - 6.0, vCenter + 6.0, tableZ, tableZ + 6.5, castChannelDark, castChannelDark);
    }

    // 2. Stationary Rear Fixed-Jaw Tower (Kurt DX6 CrossOver high-rigidity cast tower + twin rear gusset ribs)
    const towerTopZ = jawTopZ - 2.2;
    const fV0 = Math.min(fixedTowerRearV, fixedTowerFrontV);
    const fV1 = Math.max(fixedTowerRearV, fixedTowerFrontV);
    addBoxUVZ(-halfW, halfW, fV0 + 18.0, fV1, bedZ, towerTopZ, kurtBlue, kurtBlue);
    // Sloped rear DX6 CrossOver buttress & twin reinforcement ribs
    addGussetRibUVZ(-halfW, -halfW + 26.0, fV0 - 4.0, fV0 + 18.0, fV0 + 14.0, fV0 + 18.0, bedZ, towerTopZ - 2.0, kurtBlueDark);
    addGussetRibUVZ(halfW - 26.0, halfW, fV0 - 4.0, fV0 + 18.0, fV0 + 14.0, fV0 + 18.0, bedZ, towerTopZ - 2.0, kurtBlueDark);
    addGussetRibUVZ(-halfW + 26.0, halfW - 26.0, fV0 + 3.0, fV0 + 18.0, fV0 + 16.0, fV0 + 18.0, bedZ, towerTopZ - 7.0, kurtBlue);

    // 3. Sliding AngLock Movable Jaw Carriage (+V)
    const mV0 = Math.min(movCarriageRearV, movCarriageFrontV);
    const mV1 = Math.max(movCarriageRearV, movCarriageFrontV);
    const carriageTopZ = jawTopZ - 3.0;
    addBoxUVZ(-halfW + 2.0, halfW - 2.0, mV0, mV0 + 38.0, bedZ + 0.4, carriageTopZ, kurtBlue, kurtBlue);
    addGussetRibUVZ(
      -halfW + 4.0,
      halfW - 4.0,
      mV0 + 38.0,
      mV1,
      mV0 + 38.0,
      mV0 + 46.0,
      bedZ + 0.4,
      carriageTopZ - 3.0,
      kurtBlueDark
    );
    // Movable nut keel inside central channel
    addBoxUVZ(-channelHalfU + 3.0, channelHalfU - 3.0, mV0 + 4.0, mV1 - 6.0, channelFloorZ + 6.0, bedZ + 1.0, castChannelDark, castChannelDark);

    // 4. Enclosed Acme Leadscrew + Front Thrust Bearing Collar + 3/4" Hex Wrench Drive Boss (+V)
    const screwZ = bedZ - 22.0;
    addCylinderAlongV(0.0, screwZ, fV1 - 10.0, bodyFrontV + 6.0, 12.5, 14, screwSteel, true);
    // Kurt Blue front bearing boss
    addCylinderAlongV(0.0, screwZ, bodyFrontV - 2.0, bodyFrontV + 14.0, 22.0, 16, kurtBlueDark, true);
    // Silver thrust collar
    addCylinderAlongV(0.0, screwZ, bodyFrontV + 14.0, bodyFrontV + 23.0, 17.5, 16, groundWaySteel, true);
    // 6-sided 3/4" hex drive end
    addCylinderAlongV(0.0, screwZ, bodyFrontV + 23.0, bodyFrontV + 42.0, 11.0, 6, hardenedJawFace, true);

    // 5. Fixed & Movable Jaw Plates + Counterbored Socket-Head Cap Screws + Parallels / Contoured Soft-Jaw Step
    if (!isSoftJaws) {
      // Setup 1 (G54): Hardened ground steel jaw plates + precision ground parallels
      addBoxUVZ(-halfW, halfW, fixedPlateV0, fixedPlateV1, bedZ, jawTopZ, hardenedJawSteel, hardenedJawFace);
      addBoxUVZ(-halfW, halfW, movPlateV0, movPlateV1, bedZ, jawTopZ, hardenedJawSteel, hardenedJawFace);

      // Socket-head cap screws (two per jaw at U = +-49.2 mm, Z = bedZ + 22.2 mm)
      const boltZ = bedZ + 0.5 * jawH;
      for (const uBolt of [-49.2, 49.2]) {
        if (!isReversed) {
          addCylinderAlongV(uBolt, boltZ, fixedPlateV1 - 3.0, fixedPlateV1 + 0.25, 6.2, 12, blackOxideBolt, true);
          addCylinderAlongV(uBolt, boltZ, movPlateV0 - 0.25, movPlateV0 + 3.0, 6.2, 12, blackOxideBolt, true);
        } else {
          addCylinderAlongV(uBolt, boltZ, fixedPlateV0 - 0.25, fixedPlateV0 + 3.0, 6.2, 12, blackOxideBolt, true);
          addCylinderAlongV(uBolt, boltZ, movPlateV1 - 3.0, movPlateV1 + 0.25, 6.2, 12, blackOxideBolt, true);
        }
      }

      // Precision ground steel parallels resting on bedZ against fixed & movable clamping faces
      if (parH > 1.0 && !isReversed) {
        const parTopZ = Math.min(seatZ, bedZ + parH);
        const effParT = Math.min(parT, Math.max(0.8, 0.42 * (spanMax - spanMin)));
        addBoxUVZ(-halfW + 1.0, halfW - 1.0, spanMin, spanMin + effParT, bedZ, parTopZ, parallelSteel, parallelSteel);
        addBoxUVZ(-halfW + 1.0, halfW - 1.0, spanMax - effParT, spanMax, bedZ, parTopZ, parallelSteel, parallelSteel);
      }
    } else {
      // Setup 2 (G55): CNC-machined 6061-T6 Aluminum Contoured Soft Jaws!
      // Outer soft jaw blocks sit behind the clamped part span, while a CNC-milled step pocket at seatZ
      // supports the flipped Op1 part perimeter cleanly below the carrier-hat facing plane.
      const stepDepthV = Math.min(8.0, 0.35 * jawT);
      const pocketU0 = Math.max(-halfW + 4.0, crossMin - 0.4);
      const pocketU1 = Math.min(halfW - 4.0, crossMax + 0.4);

      // Base portion of soft jaws (from bedZ up to seatZ, extending inward by stepDepthV to form precision seating ledge!)
      addBoxUVZ(-halfW, halfW, fixedPlateV0, spanMin + stepDepthV, bedZ, seatZ, softJawAlum, softJawSeat);
      addBoxUVZ(-halfW, halfW, spanMax - stepDepthV, movPlateV1, bedZ, seatZ, softJawAlum, softJawSeat);

      // Upper clamping wall from seatZ to jawTopZ:
      // Back spine behind clamping face
      addBoxUVZ(-halfW, halfW, fixedPlateV0, spanMin, seatZ, jawTopZ, softJawAlum, softJawSeat);
      addBoxUVZ(-halfW, halfW, spanMax, movPlateV1, seatZ, jawTopZ, softJawAlum, softJawSeat);
      // Side locating shoulders outside [pocketU0, pocketU1] so the contoured pocket visibly cradles the flipped part!
      if (pocketU0 > -halfW + 5.0) {
        addBoxUVZ(-halfW, pocketU0, spanMin, spanMin + stepDepthV, seatZ, jawTopZ, softJawAlum, softJawSeat);
        addBoxUVZ(-halfW, pocketU0, spanMax - stepDepthV, spanMax, seatZ, jawTopZ, softJawAlum, softJawSeat);
      }
      if (pocketU1 < halfW - 5.0) {
        addBoxUVZ(pocketU1, halfW, spanMin, spanMin + stepDepthV, seatZ, jawTopZ, softJawAlum, softJawSeat);
        addBoxUVZ(pocketU1, halfW, spanMax - stepDepthV, spanMax, seatZ, jawTopZ, softJawAlum, softJawSeat);
      }

      const boltZ = bedZ + 0.46 * jawH;
      for (const uBolt of [-49.2, 49.2]) {
        addCylinderAlongV(uBolt, boltZ, fixedPlateV0 - 0.25, fixedPlateV0 + 3.0, 6.2, 12, blackOxideBolt, true);
        addCylinderAlongV(uBolt, boltZ, movPlateV1 - 3.0, movPlateV1 + 0.25, 6.2, 12, blackOxideBolt, true);
      }
    }

    return {
      positions: new Float32Array(posArr),
      normals: new Float32Array(nrmArr),
      colors: new Float32Array(colArr),
      vertexCount: posArr.length / 3,
      setupInfo: st,
      workholding: wh,
    };
  }

  // Extrude 3D segments into 4-sided 3D prism tubes (24 vertices / segment) so path width scales cleanly on all GPUs.
  // When zFlipSum is provided, Op2 (setup_index === 2, G55) moves are stored in workpiece-attached G54 coordinates
  // so flipping the workpiece with mat4FlipX180(zFlipSum) during Op2 aligns stock, CAD, and toolpath ribbons!
  // Optional segmentFilterFn(seg, idx) lets users visualize any single or multi-operation subset cleanly.
  function buildThickPathTubeGeometry(segments, wantRapid, radiusMm, colorFn, zFlipSum, segmentFilterFn) {
    const matchedIndices = [];
    for (let i = 0; i < segments.length; i++) {
      const isR = segments[i].kind === "rapid";
      if (isR === wantRapid) {
        if (segmentFilterFn && !segmentFilterFn(segments[i], i)) {
          continue;
        }
        const st = segments[i].start;
        const en = segments[i].end;
        const len = Math.hypot(en[0] - st[0], en[1] - st[1], en[2] - st[2]);
        if (len > 1e-5) matchedIndices.push(i);
      }
    }

    const nSegs = matchedIndices.length;
    const vertsPerSeg = 24; // 4 quads * 2 tris * 3 verts
    const positions = new Float32Array(nSegs * vertsPerSeg * 3);
    const normals = new Float32Array(nSegs * vertsPerSeg * 3);
    const colors = new Float32Array(nSegs * vertsPerSeg * 3);
    const hasFlip = Number.isFinite(zFlipSum);
    let o = 0;

    for (let m = 0; m < nSegs; m++) {
      const segIdx = matchedIndices[m];
      const s = segments[segIdx];
      const isOp2 = hasFlip && Number(s.setup_index || 1) === 2;
      const ax = s.start[0];
      const ay = isOp2 ? -s.start[1] : s.start[1];
      const az = isOp2 ? zFlipSum - s.start[2] : s.start[2];
      const bx = s.end[0];
      const by = isOp2 ? -s.end[1] : s.end[1];
      const bz = isOp2 ? zFlipSum - s.end[2] : s.end[2];
      let tx = bx - ax, ty = by - ay, tz = bz - az;
      const len = Math.hypot(tx, ty, tz) || 1.0;
      tx /= len; ty /= len; tz /= len;

      // Construct orthonormal basis (U, V) perpendicular to T
      let ux, uy, uz;
      if (Math.abs(tz) < 0.92) {
        ux = -ty; uy = tx; uz = 0.0;
      } else {
        ux = 0.0; uy = -tz; uz = ty;
      }
      const uLen = Math.hypot(ux, uy, uz) || 1.0;
      ux /= uLen; uy /= uLen; uz /= uLen;
      const vx = ty * uz - tz * uy;
      const vy = tz * ux - tx * uz;
      const vz = tx * uy - ty * ux;

      const rgb = colorFn(s);
      const r = radiusMm;

      // 4 corner offsets around axis: (+U), (+V), (-U), (-V)
      const dirs = [
        [ux, uy, uz],
        [vx, vy, vz],
        [-ux, -uy, -uz],
        [-vx, -vy, -vz],
      ];

      for (let side = 0; side < 4; side++) {
        const d0 = dirs[side];
        const d1 = dirs[(side + 1) & 3];
        let fnx = d0[0] + d1[0], fny = d0[1] + d1[1], fnz = d0[2] + d1[2];
        const fnL = Math.hypot(fnx, fny, fnz) || 1.0;
        fnx /= fnL; fny /= fnL; fnz /= fnL;

        const p00x = ax + r * d0[0], p00y = ay + r * d0[1], p00z = az + r * d0[2];
        const p01x = ax + r * d1[0], p01y = ay + r * d1[1], p01z = az + r * d1[2];
        const p11x = bx + r * d1[0], p11y = by + r * d1[1], p11z = bz + r * d1[2];
        const p10x = bx + r * d0[0], p10y = by + r * d0[1], p10z = bz + r * d0[2];

        positions[o] = p00x; positions[o + 1] = p00y; positions[o + 2] = p00z;
        positions[o + 3] = p10x; positions[o + 4] = p10y; positions[o + 5] = p10z;
        positions[o + 6] = p11x; positions[o + 7] = p11y; positions[o + 8] = p11z;
        positions[o + 9] = p00x; positions[o + 10] = p00y; positions[o + 11] = p00z;
        positions[o + 12] = p11x; positions[o + 13] = p11y; positions[o + 14] = p11z;
        positions[o + 15] = p01x; positions[o + 16] = p01y; positions[o + 17] = p01z;

        for (let k = 0; k < 18; k += 3) {
          normals[o + k] = fnx; normals[o + k + 1] = fny; normals[o + k + 2] = fnz;
          colors[o + k] = rgb[0]; colors[o + k + 1] = rgb[1]; colors[o + k + 2] = rgb[2];
        }
        o += 18;
      }
    }

    return {
      positions,
      normals,
      colors,
      segIndices: matchedIndices,
      vertsPerSeg,
      totalVerts: nSegs * vertsPerSeg,
    };
  }

  // --- Build Static Machine Table Grid + G54 Triad + Billet Wireframe Lines ---
  function buildSceneGuideLines(stockMin, stockMax, workholding) {
    const verts = [];
    const cols = [];
    function addSeg(a, b, rgba) {
      verts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      cols.push(rgba[0], rgba[1], rgba[2], rgba[3], rgba[0], rgba[1], rgba[2], rgba[3]);
    }

    const spanX = Math.abs(stockMax[0] - stockMin[0]);
    const spanY = Math.abs(stockMax[1] - stockMin[1]);
    const maxSpan = Math.max(180.0, spanX, spanY);
    const gridStep = maxSpan > 220.0 ? 25 : 20;
    const gridExtent = Math.max(240, Math.ceil((maxSpan * 1.35) / gridStep) * gridStep);
    const setup1 = workholding && workholding.setups && workholding.setups["1"];
    const zFloor =
      setup1 && Number.isFinite(Number(setup1.table_z_mm))
        ? Number(setup1.table_z_mm) - 0.15
        : stockMin[2] - 115.5;
    for (let g = -gridExtent; g <= gridExtent; g += gridStep) {
      const c = g === 0 ? [0.26, 0.33, 0.43, 0.65] : [0.18, 0.23, 0.30, 0.42];
      addSeg([-gridExtent, g, zFloor], [gridExtent, g, zFloor], c);
      addSeg([g, -gridExtent, zFloor], [g, gridExtent, zFloor], c);
    }

    const axisLen = Math.max(10.0, Math.min(28.0, Math.max(50.0, spanX, spanY) * 0.16));
    const zTriad = Math.max(0.08, stockMax[2] + 0.08);
    addSeg([0, 0, zTriad], [axisLen, 0, zTriad], [1.0, 0.42, 0.42, 1.0]);
    addSeg([0, 0, zTriad], [0, axisLen, zTriad], [0.45, 0.92, 0.52, 1.0]);
    addSeg([0, 0, zTriad], [0, 0, zTriad + axisLen], [0.42, 0.72, 1.0, 1.0]);

    const gridVertexCount = verts.length / 3;

    const [x0, y0, z0] = stockMin;
    const [x1, y1, z1] = stockMax;
    const bc = [0.68, 0.78, 0.90, 0.68];
    const corners = [
      [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    for (const [i, j] of edges) {
      addSeg(corners[i], corners[j], bc);
    }

    // Highlight the Kurt DX6 jaw-bite keep-out line around the billet perimeter in Setup 1
    if (setup1 && Number.isFinite(Number(setup1.jaw_top_z_mm))) {
      const zGrip = Number(setup1.jaw_top_z_mm);
      if (zGrip > z0 + 0.1 && zGrip < z1 - 0.1) {
        const gc = [0.98, 0.68, 0.18, 0.62];
        addSeg([x0, y0, zGrip], [x1, y0, zGrip], gc);
        addSeg([x1, y0, zGrip], [x1, y1, zGrip], gc);
        addSeg([x1, y1, zGrip], [x0, y1, zGrip], gc);
        addSeg([x0, y1, zGrip], [x0, y0, zGrip], gc);
      }
    }

    return {
      positions: new Float32Array(verts),
      colors: new Float32Array(cols),
      gridCount: gridVertexCount,
      billetOffset: gridVertexCount,
      billetCount: verts.length / 3 - gridVertexCount,
    };
  }

  function defaultCamDistance(caseData) {
    if (!caseData || !caseData.stock_min_mm || !caseData.stock_max_mm) return 145.0;
    const dx = Number(caseData.stock_max_mm[0]) - Number(caseData.stock_min_mm[0]);
    const dy = Number(caseData.stock_max_mm[1]) - Number(caseData.stock_min_mm[1]);
    const dz = Number(caseData.stock_max_mm[2]) - Number(caseData.stock_min_mm[2]);
    const diag = Math.hypot(dx, dy, dz);
    return Math.max(125.0, Math.min(620.0, diag * 1.72));
  }

  // --- Resolve or Synthesize Haas WIPS-R (OMP40-2 + OTS) Setup Sheet & Per-Tool Schedule ---
  function computeSafeSideTouchZMm(zTopMm, sideDepthMm, jawTopZMm, stockMinZMm) {
    let desired = Number(zTopMm) - Math.abs(Number(sideDepthMm || 3.5));
    const floors = [];
    if (Number.isFinite(jawTopZMm)) floors.push(Number(jawTopZMm) + 0.55);
    if (Number.isFinite(stockMinZMm)) floors.push(Number(stockMinZMm) + 0.50);
    if (floors.length > 0) {
      desired = Math.max(desired, Math.max.apply(null, floors));
    }
    return Number(Math.min(Number(zTopMm) - 0.60, desired).toFixed(3));
  }

  function classifyOtsModeForTool(etype, diameterMm, otsEnabled) {
    const et = String(etype || "flat end mill").toLowerCase();
    if (et === "probe") {
      return {
        mode: "spindle_probe_master",
        label: "OMP40-2 Master WCS Probe (P9832)",
      };
    }
    if (!otsEnabled) {
      return { mode: "manual_offset", label: "Manual Length / Radius Offset" };
    }
    const pointTypes = [
      "drill",
      "spot drill",
      "center drill",
      "countersink",
      "chamfer mill",
      "engraving",
      "tap right hand",
      "tap left hand",
      "reamer",
    ];
    if (pointTypes.indexOf(et) >= 0 || Number(diameterMm || 0) < 1.5) {
      return {
        mode: "ots_centerline_length",
        label: "OTS Non-Rotating Centerline Length (B1)",
      };
    }
    return {
      mode: "ots_length_and_diameter",
      label: "OTS Length + Rotating Diameter (B3)",
    };
  }

  function resolveCaseSetupSheet(caseData, variant) {
    if (!caseData) return null;
    if (
      caseData.setup_sheet &&
      Array.isArray(caseData.setup_sheet.setups) &&
      caseData.setup_sheet.setups.length > 0 &&
      Array.isArray(caseData.setup_sheet.tools_schedule)
    ) {
      return caseData.setup_sheet;
    }

    const smin = (caseData.stock_min_mm || [-25, -20, -10]).map(Number);
    const smax = (caseData.stock_max_mm || [25, 20, 0]).map(Number);
    const pmin = (caseData.pocket_min_mm || [smin[0] + 2, smin[1] + 2, smin[2] + 2]).map(Number);
    const pmax = (caseData.pocket_max_mm || [smax[0] - 2, smax[1] - 2, smax[2]]).map(Number);
    const lx = Number((smax[0] - smin[0]).toFixed(2));
    const ly = Number((smax[1] - smin[1]).toFixed(2));
    const lz = Number((smax[2] - smin[2]).toFixed(2));
    const cx = Number((0.5 * (smin[0] + smax[0])).toFixed(3));
    const cy = Number((0.5 * (smin[1] + smax[1])).toFixed(3));

    const wh = caseData.workholding || resolveCaseWorkholding(caseData);
    const derivedOps = deriveCaseOperations(caseData, variant || (caseData.variants && caseData.variants[0]));
    const hasOp2 = Boolean(
      (wh && wh.has_op2_soft_jaws) || derivedOps.some((o) => Number(o.setupIndex || 1) >= 2)
    );
    const clampAxis = String((wh && wh.clamping_axis) || "Y").toUpperCase();
    const s1Wh = (wh && wh.setups && wh.setups["1"]) || {};
    const s2Wh = (wh && wh.setups && wh.setups["2"]) || {};
    const jawTopS1 = Number(
      s1Wh.jaw_top_z_mm !== undefined ? s1Wh.jaw_top_z_mm : smin[2] + 2.0
    );
    const zSideS1 = computeSafeSideTouchZMm(0.0, 3.5, jawTopS1, smin[2]);
    const rStylus = 3.0; // Ø6.0 mm Renishaw OMP40-2 ruby ball

    const fixedRef =
      clampAxis === "Y"
        ? "Fixed Rear Jaw (+Y / Stationary Vise Body) — set X workstop on left (-X) billet face."
        : "Fixed Left Jaw (-X / Stationary Vise Body) — set Y workstop on front (-Y) billet face.";

    const setups = [
      {
        setup_index: 1,
        work_offset: "G54",
        stage_badge: "INITIAL PROBE · STEP 1",
        title: "Setup 1 (G54) — Raw Billet Clamping & OMP40-2 Pre-Start Position",
        vise_configuration:
          String(s1Wh.label || "") ||
          "Kurt DX6 Hardened Jaws + " + Number(s1Wh.parallel_height_mm || 42.45).toFixed(2) + " mm Parallels",
        clamping_axis: clampAxis,
        jaw_opening_mm: Number(s1Wh.jaw_opening_mm !== undefined ? s1Wh.jaw_opening_mm : clampAxis === "Y" ? ly : lx),
        grip_z_mm: Number(s1Wh.grip_z_mm !== undefined ? s1Wh.grip_z_mm : 2.0),
        parallel_height_mm: Number(s1Wh.parallel_height_mm !== undefined ? s1Wh.parallel_height_mm : 42.45),
        jaw_top_z_mm: Number(jawTopS1.toFixed(3)),
        fixed_jaw_reference: fixedRef,
        wcs_datum_description:
          "Top-Center of Raw Stock Billet (X = 0.000, Y = 0.000, Z = 0.000 at unmachined top face)",
        probe_start_xyz_mm: [cx, cy, 10.0],
        probe_start_xyz_in: [
          Number((cx / INCH_MM).toFixed(4)),
          Number((cy / INCH_MM).toFixed(4)),
          Number((10.0 / INCH_MM).toFixed(4)),
        ],
        xy_jog_tolerance_mm: 5.0,
        z_jog_tolerance_mm: 3.0,
        probed_span_xy_mm: [lx, ly],
        side_touch_z_mm: zSideS1,
        radial_clearance_mm: 8.0,
        touch_points_mm: [
          { id: "Z_TOP", label: "1. Top Face Z0 Datum", xyz_mm: [cx, cy, 0.0] },
          { id: "X_NEG", label: "2. -X Billet Wall", xyz_mm: [Number((smin[0] - rStylus).toFixed(3)), cy, zSideS1] },
          { id: "X_POS", label: "3. +X Billet Wall", xyz_mm: [Number((smax[0] + rStylus).toFixed(3)), cy, zSideS1] },
          { id: "Y_NEG", label: "4. -Y Billet Wall", xyz_mm: [cx, Number((smin[1] - rStylus).toFixed(3)), zSideS1] },
          { id: "Y_POS", label: "5. +Y Billet Wall", xyz_mm: [cx, Number((smax[1] + rStylus).toFixed(3)), zSideS1] },
        ],
        checklist: [
          "Verify Renishaw OMP40-2 (T99, Ø6.0 mm ruby ball) in spindle/ATC and Renishaw OTS clear on right (+X) table.",
          "Seat " +
            lx.toFixed(1) +
            " × " +
            ly.toFixed(1) +
            " × " +
            lz.toFixed(1) +
            " mm billet on " +
            Number(s1Wh.parallel_height_mm || 42.45).toFixed(2) +
            " mm parallels in Kurt DX6 (" +
            Number(s1Wh.grip_z_mm || 2.0).toFixed(1) +
            " mm grip bite); tap down flush.",
          "Handle-jog T99 OMP40-2 ruby tip over billet top-center within ±5 mm XY and Z = +10 mm (+0.39 in) above top surface; set rough initial G54 XYZ.",
          "Press Cycle Start: OTS sets length (H) & rotating diameter (D) for cutting tools, then OMP40-2 probes exact G54 Z0 and XY web center before Op1 cutting.",
        ],
      },
    ];

    if (hasOp2) {
      const jawTopS2 = Number(s2Wh.jaw_top_z_mm !== undefined ? s2Wh.jaw_top_z_mm : -3.2);
      const zSideS2 = computeSafeSideTouchZMm(0.0, 3.5, jawTopS2, smin[2]);
      setups.push({
        setup_index: 2,
        work_offset: "G55",
        stage_badge: "POST-FLIP RE-PROBE · AFTER M00",
        title: "Setup 2 (G55) — 180° Part Flip in Soft Jaws & OMP40-2 Re-Probing",
        vise_configuration:
          String(s2Wh.label || "") || "Kurt DX6 Machined 6061-T6 Contoured Soft Jaws",
        clamping_axis: clampAxis,
        jaw_opening_mm: Number(s2Wh.jaw_opening_mm !== undefined ? s2Wh.jaw_opening_mm : clampAxis === "Y" ? ly : lx),
        grip_z_mm: Number(s2Wh.grip_z_mm !== undefined ? s2Wh.grip_z_mm : 4.0),
        seat_z_mm: Number(s2Wh.seat_z_mm !== undefined ? s2Wh.seat_z_mm : -6.0),
        jaw_top_z_mm: Number(jawTopS2.toFixed(3)),
        fixed_jaw_reference:
          "Contoured G55 6061-T6 Soft Jaws — flip part 180° around X-axis (Y -> -Y, Z -> -Z) so carrier hat faces +Z.",
        wcs_datum_description:
          "Top-Center of Flipped Carrier Hat / Part in G55 Soft Jaws (X = 0.000, Y = 0.000, Z = 0.000 at flipped top)",
        probe_start_xyz_mm: [cx, cy, 10.0],
        probe_start_xyz_in: [
          Number((cx / INCH_MM).toFixed(4)),
          Number((cy / INCH_MM).toFixed(4)),
          Number((10.0 / INCH_MM).toFixed(4)),
        ],
        xy_jog_tolerance_mm: 5.0,
        z_jog_tolerance_mm: 3.0,
        probed_span_xy_mm: [lx, ly],
        side_touch_z_mm: zSideS2,
        radial_clearance_mm: 8.0,
        touch_points_mm: [
          { id: "Z_TOP", label: "1. Flipped Top Z0 Datum", xyz_mm: [cx, cy, 0.0] },
          { id: "X_NEG", label: "2. -X Flipped Hat Wall", xyz_mm: [Number((smin[0] - rStylus).toFixed(3)), cy, zSideS2] },
          { id: "X_POS", label: "3. +X Flipped Hat Wall", xyz_mm: [Number((smax[0] + rStylus).toFixed(3)), cy, zSideS2] },
          { id: "Y_NEG", label: "4. -Y Flipped Hat Wall", xyz_mm: [cx, Number((smin[1] - rStylus).toFixed(3)), zSideS2] },
          { id: "Y_POS", label: "5. +Y Flipped Hat Wall", xyz_mm: [cx, Number((smax[1] + rStylus).toFixed(3)), zSideS2] },
        ],
        checklist: [
          "At M00 program stop, open Kurt DX6 jaws, clear chips with air blast, and clean G55 contoured 6061 soft jaws.",
          "Flip Op1-machined part 180° about the X-axis (unmachined Op1 grip hat faces +Z) and torque into G55 soft-jaw pocket.",
          "Verify G55 rough seed is within ±5 mm of flipped center (side touches clear soft jaws at Z = " +
            zSideS2.toFixed(2) +
            " mm > jaw top " +
            jawTopS2.toFixed(2) +
            " mm).",
          "Press Cycle Start: T99 OMP40-2 automatically re-probes exact G55 Z0 and XY center before facing off the carrier hat and finishing Op2.",
        ],
      });
    }

    // Build combined per-tool list (T99 OMP40-2 Spindle Probe + all cutting tools)
    const opsByTnum = new Map();
    for (const op of derivedOps) {
      const tn = Number(op.toolNumber || 1);
      if (!opsByTnum.has(tn)) opsByTnum.set(tn, []);
      opsByTnum.get(tn).push("Op" + op.opIndex + " (" + op.workOffset + ")");
    }

    const rawToolMap = new Map();
    rawToolMap.set(99, {
      tool_number: 99,
      tool_id: "probe_omp40_2_mm_6mm",
      entity_type: "probe",
      diameter_mm: 6.0,
      flutes: 1,
      flute_length_mm: 6.0,
      stickout_mm: 50.0,
      nominal_label: "Renishaw OMP40-2 Optical Spindle Probe (Ø6.0 mm Ruby Stylus, 50 mm Stem)",
      holder_description: "BT30 / CAT40 Optical Probe Shank",
    });

    const tbl = caseData.tools_table;
    if (tbl && typeof tbl === "object") {
      const entries = Array.isArray(tbl) ? tbl : Object.values(tbl);
      for (const item of entries) {
        if (!item) continue;
        const tn = Number(item.tool_number || item.t_number || 1);
        rawToolMap.set(tn, item);
      }
    }
    if (rawToolMap.size <= 1 && caseData.tool) {
      const t0 = caseData.tool;
      const tn0 = Number(t0.tool_number || 1);
      const d0 = Number(t0.diameter_mm || 2.0 * Number(t0.cutter_radius_mm || 2.5));
      rawToolMap.set(tn0, {
        tool_number: tn0,
        tool_id: t0.tool_id || "T" + tn0,
        entity_type: t0.entity_type || "flat end mill",
        diameter_mm: d0,
        flutes: Number(t0.flutes || 3),
        flute_length_mm: Number(t0.flute_length_mm || 12.0),
        stickout_mm: Number(t0.stickout_mm || t0.holder_bottom_mm || 24.0),
        nominal_label:
          t0.nominal_label || "Ø" + d0.toFixed(2) + " mm Solid Carbide End Mill",
        holder_description: t0.holder_description || "BT30 / CAT40 ER Collet Chuck",
      });
    }

    const sortedTnums = Array.from(rawToolMap.keys()).sort((a, b) => {
      if (a === 99) return -1;
      if (b === 99) return 1;
      return a - b;
    });

    const toolsSchedule = [];
    for (const tn of sortedTnums) {
      const rec = rawToolMap.get(tn);
      const dMm = Number(rec.diameter_mm || 2.0 * Number(rec.cutter_radius_mm || 3.0));
      const stickMm = Number(rec.stickout_mm || rec.holder_bottom_mm || 30.0);
      const fluteMm = Number(rec.flute_length_mm || 0.5 * stickMm);
      const flutes = Math.max(1, Number(rec.flutes || (tn === 99 ? 1 : 3)));
      const etype = String(rec.entity_type || (tn === 99 ? "probe" : "flat end mill"));
      const lbl = String(rec.nominal_label || rec.label || "Tool T" + tn);
      const hld = String(rec.holder_description || rec.holder_id || "BT30 / CAT40 Collet");
      const otsInfo = classifyOtsModeForTool(etype, dMm, true);
      const macroStr =
        otsInfo.mode === "spindle_probe_master"
          ? "T" + tn + " Master OMP40-2 (G65 P9832 / P9811 / P9812)"
          : "G65 P9995 A0. " +
            (otsInfo.mode === "ots_length_and_diameter" ? "B3." : "B1.") +
            " C2. T" +
            tn +
            ". E" +
            stickMm.toFixed(1) +
            " D" +
            dMm.toFixed(2);
      const usedOps =
        opsByTnum.get(tn) ||
        (tn === 99 ? (hasOp2 ? ["Pre-Op1 (G54)", "Post-Flip (G55)"] : ["Pre-Op1 (G54)"]) : ["Op1 (G54)"]);
      toolsSchedule.push({
        tool_number: tn,
        offset_registers: "H" + tn + (otsInfo.mode === "ots_length_and_diameter" ? " / D" + tn : ""),
        tool_id: String(rec.tool_id || "t" + tn),
        label: lbl,
        entity_type: etype,
        diameter_mm: Number(dMm.toFixed(3)),
        diameter_in: Number((dMm / INCH_MM).toFixed(4)),
        flutes,
        flute_length_mm: Number(fluteMm.toFixed(2)),
        stickout_mm: Number(stickMm.toFixed(2)),
        stickout_in: Number((stickMm / INCH_MM).toFixed(3)),
        holder_label: hld,
        ots_mode: otsInfo.mode,
        ots_mode_label: otsInfo.label,
        ots_macro_preview: macroStr,
        used_in_ops: usedOps,
      });
    }

    return {
      schema: "diffcam-setup-sheet/1",
      part_id: String(caseData.case_id || "part"),
      part_title: String(caseData.part_title || caseData.case_id || "Part"),
      program_number: 2001,
      stock_material_id: String(caseData.stock_material_id || "al_6061_t6"),
      primary_dialect: "haas_ngc",
      probing_enabled: true,
      probe_after_flip: hasOp2,
      ots_enabled: true,
      probe_package: {
        profile_id: "haas_wips_r",
        package_label: "Haas WIPS-R (Renishaw OMP40-2 Spindle Probe + OTS Tool Setter)",
        package_url: "https://haascnc.com/productivity/probe-system/wips-r",
        spindle_probe: {
          model_id: "renishaw_omp40_2",
          label: "Renishaw OMP40-2 Optical Spindle Probe",
          tool_number: 99,
          stylus_diameter_mm: 6.0,
          stylus_length_mm: 50.0,
          body_diameter_mm: 40.0,
          pre_position_z_mm: 10.0,
          xy_jog_tolerance_mm: 5.0,
        },
        tool_setter: {
          model_id: "renishaw_ots",
          label: "Renishaw OTS Optical Table Tool Setter",
          stylus_disc_diameter_mm: 12.7,
        },
      },
      stock_dims_mm: [lx, ly, lz],
      stock_min_mm: smin,
      stock_max_mm: smax,
      part_dims_mm: [
        Number((pmax[0] - pmin[0]).toFixed(2)),
        Number((pmax[1] - pmin[1]).toFixed(2)),
        Number((pmax[2] - pmin[2]).toFixed(2)),
      ],
      has_op2_flip: hasOp2,
      setups,
      tools_schedule: toolsSchedule,
    };
  }

  function formatSetupSheetMarkdown(sheet) {
    if (!sheet) return "";
    const out = [];
    const progStr = "O" + String(Number(sheet.program_number || 2001)).padStart(5, "0");
    const dims = sheet.stock_dims_mm || [0, 0, 0];
    const pkg = (sheet.probe_package && sheet.probe_package.package_label) || "Haas WIPS-R (OMP40-2 + OTS)";
    out.push("# CNC SETUP SHEET — " + progStr + " (" + sheet.part_title + ")");
    out.push("- Part ID: " + sheet.part_id + " | Material: " + String(sheet.stock_material_id || "").toUpperCase());
    out.push("- Raw Billet: " + dims[0] + " × " + dims[1] + " × " + dims[2] + " mm");
    out.push("- Probing System: " + pkg);
    out.push("");
    for (const st of sheet.setups || []) {
      const pStart = st.probe_start_xyz_mm || [0, 0, 10];
      out.push("## " + st.title);
      out.push("- Work Offset: " + st.work_offset + " | Vise: " + st.vise_configuration);
      out.push("- Datum Origin: " + st.wcs_datum_description);
      out.push(
        "- Pre-Run OMP40-2 Position: X=" +
          pStart[0] +
          " mm, Y=" +
          pStart[1] +
          " mm (±" +
          st.xy_jog_tolerance_mm +
          " mm), Z=+" +
          pStart[2] +
          " mm | Side Web Touch Z=" +
          st.side_touch_z_mm +
          " mm (Jaw Top Z=" +
          st.jaw_top_z_mm +
          " mm)"
      );
      for (let i = 0; i < (st.checklist || []).length; i++) {
        out.push("  " + (i + 1) + ". " + st.checklist[i]);
      }
      out.push("");
    }
    out.push("## Tool Crib & Renishaw OTS Calibration Schedule");
    for (const t of sheet.tools_schedule || []) {
      out.push(
        "- T" +
          t.tool_number +
          " (" +
          t.offset_registers +
          "): " +
          t.label +
          " | Ø" +
          t.diameter_mm +
          " mm (" +
          t.diameter_in +
          " in) | Stickout " +
          t.stickout_mm +
          " mm | " +
          t.ots_macro_preview +
          " | Ops: " +
          (t.used_in_ops || []).join(", ")
      );
    }
    return out.join("\n");
  }

  function createPaneSlot(gl) {
    return {
      variant: null,
      roleTitle: "",
      isBest: false,
      dexelSim: null,
      liveStockPos: gl.createBuffer(),
      liveStockNrm: gl.createBuffer(),
      liveStockCol: gl.createBuffer(),
      liveStockCount: 0,
      staticStockPos: gl.createBuffer(),
      staticStockNrm: gl.createBuffer(),
      staticStockCol: gl.createBuffer(),
      staticStockCount: 0,
      cutPos: gl.createBuffer(),
      cutNrm: gl.createBuffer(),
      cutCol: gl.createBuffer(),
      cutSegIndices: [],
      cutVertsPerSeg: 24,
      cutTotalVerts: 0,
      rapidPos: gl.createBuffer(),
      rapidNrm: gl.createBuffer(),
      rapidCol: gl.createBuffer(),
      rapidSegIndices: [],
      rapidVertsPerSeg: 24,
      rapidTotalVerts: 0,
      activePos: gl.createBuffer(),
      activeNrm: gl.createBuffer(),
      activeCol: gl.createBuffer(),
      activeVerts: 0,
      toolPos: [0.0, 0.0, 6.0],
      activeSegIndex: -1,
      activeSegAlpha: 0.0,
      activeSeg: null,
      activeToolGpu: null,
      activeToolLabel: "T1",
      activeSetupIndex: 1,
      activeWorkOffset: "G54",
      activeOpKey: "",
      activeOpLabel: "Op 1",
      activeGcodeLine: 1,
      finished: false,
    };
  }

  // --- WebGL1 Renderer Setup (Explicit Attribute Locations 0, 1, 2) ---
  function initWebGL(canvas) {
    const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    if (!gl) {
      throw new Error("WebGL is unavailable in this browser session.");
    }

    function compile(type, src) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error("WebGL shader compilation failed.");
      }
      return sh;
    }

    function link(vsSrc, fsSrc, bindings) {
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
      for (const [loc, name] of bindings) {
        gl.bindAttribLocation(prog, loc, name);
      }
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error("WebGL program link failed.");
      }
      return prog;
    }

    const meshProg = link(
      `
      attribute vec3 aPos;
      attribute vec3 aNormal;
      attribute vec3 aColor;
      uniform mat4 uMVP;
      uniform mat4 uModel;
      uniform vec3 uBaseColor;
      uniform float uUseVertColor;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying vec3 vColor;
      void main() {
        vec4 wp = uModel * vec4(aPos, 1.0);
        vWorldPos = wp.xyz;
        vNormal = normalize(mat3(uModel[0].xyz, uModel[1].xyz, uModel[2].xyz) * aNormal);
        vColor = mix(uBaseColor, aColor, uUseVertColor);
        gl_Position = uMVP * wp;
      }
      `,
      `
      precision mediump float;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying vec3 vColor;
      uniform vec3 uEye;
      uniform float uAlpha;
      uniform float uUnlitMix;
      void main() {
        vec3 N = normalize(vNormal);
        vec3 V = normalize(uEye - vWorldPos);
        if (dot(N, V) < 0.0) {
          N = -N;
        }
        vec3 L1 = normalize(vec3(0.52, -0.62, 0.85));
        vec3 L2 = normalize(vec3(-0.68, 0.48, 0.35));
        float diff1 = max(dot(N, L1), 0.0);
        float diff2 = max(dot(N, L2), 0.0);
        vec3 H = normalize(L1 + V);
        float spec = pow(max(dot(N, H), 0.0), 36.0) * 0.24 * (1.0 - uUnlitMix);
        float amb = 0.32 + 0.08 * (0.5 * N.z + 0.5);
        vec3 lit = vColor * (amb + 0.56 * diff1 + 0.20 * diff2) + vec3(spec);
        vec3 outRgb = mix(lit, vColor, uUnlitMix);
        gl_FragColor = vec4(outRgb, uAlpha);
      }
      `,
      [
        [0, "aPos"],
        [1, "aNormal"],
        [2, "aColor"],
      ]
    );

    const lineProg = link(
      `
      attribute vec3 aPos;
      attribute vec4 aColor;
      uniform mat4 uMVP;
      varying vec4 vColor;
      void main() {
        vColor = aColor;
        gl_Position = uMVP * vec4(aPos, 1.0);
      }
      `,
      `
      precision mediump float;
      varying vec4 vColor;
      void main() {
        gl_FragColor = vColor;
      }
      `,
      [
        [0, "aPos"],
        [1, "aColor"],
      ]
    );

    return {
      gl,
      mesh: {
        prog: meshProg,
        uMVP: gl.getUniformLocation(meshProg, "uMVP"),
        uModel: gl.getUniformLocation(meshProg, "uModel"),
        uBaseColor: gl.getUniformLocation(meshProg, "uBaseColor"),
        uUseVertColor: gl.getUniformLocation(meshProg, "uUseVertColor"),
        uEye: gl.getUniformLocation(meshProg, "uEye"),
        uAlpha: gl.getUniformLocation(meshProg, "uAlpha"),
        uUnlitMix: gl.getUniformLocation(meshProg, "uUnlitMix"),
      },
      line: {
        prog: lineProg,
        uMVP: gl.getUniformLocation(lineProg, "uMVP"),
      },
      shared: {
        targetPos: gl.createBuffer(),
        targetNrm: gl.createBuffer(),
        toolPos: gl.createBuffer(),
        toolNrm: gl.createBuffer(),
        toolCol: gl.createBuffer(),
        guidePos: gl.createBuffer(),
        guideCol: gl.createBuffer(),
        vise1Pos: gl.createBuffer(),
        vise1Nrm: gl.createBuffer(),
        vise1Col: gl.createBuffer(),
        vise2Pos: gl.createBuffer(),
        vise2Nrm: gl.createBuffer(),
        vise2Col: gl.createBuffer(),
      },
      toolMeshCache: new Map(),
      panes: [createPaneSlot(gl), createPaneSlot(gl), createPaneSlot(gl)],
    };
  }

  // --- Application State ---
  const state = {
    manifest: null,
    catalog: DEFAULT_FS_CATALOG,
    currentCase: null,
    currentCaseUrl: "",
    currentVariant: null,
    activeTab: "chipload",
    stlCache: new Map(),
    loadSeq: 0,

    // Tool Library, Unit System ("inches" | "millimeters" | "both"), & Available Tool Crib
    toolCatalog: null,
    toolLibrary: [],
    toolLibraryById: new Map(),
    starterCribs: {},
    unitSystem: "inches",
    selectedToolIds: new Set(),
    cribSearch: "",
    cribCategory: "all",

    // Right-Side G-Code Tab State
    gcodeDialect: "haas_ngc",
    gcodeAutoFollow: true,
    gcodeProgramCache: null,
    lastHighlightedLine: -1,

    // Interactive Server Session & Upload Modal State (CSRF kept strictly in-memory)
    csrfToken: "",
    serverInteractive: false,
    exampleParts: [],
    uploadSourceMode: "upload",
    pendingUploadFile: null,
    pendingUploadBase64: "",
    machiningBusy: false,

    // Layout & Fair / Optimal-Chipload Physics Policy State
    layoutMode: "single", // "single" | "sbs"
    speedPolicy: "matched", // DEFAULT: Same Speed Across Methods ("matched" | "optimal_chipload")
    aggressiveness: 1.0, // 0.40 .. 1.80
    stockMaterialId: "al_6061_t6",
    toolMaterialId: "carbide_coated",
    baselinePhysics: null,

    // Appearance & Thick Ribbon Colorization State
    stockMode: "live_alloy",
    stockPalette: "green",
    stockAlpha: 1.0,
    pathColorMode: "chipload",
    pathWidthMm: 0.28,
    tightRange: { min: 0.0, max: 1.0, span: 1.0, constant: false },

    targetCount: 0,
    toolCount: 0,
    vise1Count: 0,
    vise2Count: 0,
    workholding: null,
    guideGridCount: 0,
    guideBilletOffset: 0,
    guideBilletCount: 0,

    // Multi-Operation Visualization & Filter State
    operations: [],
    opByKey: new Map(),
    opColorIndexMap: new Map(),
    opFilterMode: "all", // "all" | "custom"
    selectedOpKeys: new Set(),
    activeOpKey: "",
    setupSheetActiveIdx: 1,

    playing: true,
    firstBoot: true,
    playTimeS: 0.0,
    totalDurationS: 0.0,
    speed: 2.0,
    lastFrameMs: 0.0,

    cam: {
      yaw: -0.78,
      pitch: 0.62,
      distance: 108.0,
      target: [0.0, 0.0, -2.5],
      dragging: false,
      panMode: false,
      lastX: 0,
      lastY: 0,
    },
  };

  let renderer = null;

  function uploadBuffer(gl, buf, float32Data, dynamic) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, float32Data, dynamic ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
  }

  async function fetchJson(url) {
    const resp = await fetch(url, {
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    });
    if (!resp.ok) {
      throw new Error("Failed to load JSON asset (" + resp.status + ").");
    }
    return resp.json();
  }

  async function fetchParsedStl(url, tintRgb) {
    if (state.stlCache.has(url)) {
      return state.stlCache.get(url);
    }
    const resp = await fetch(url, { credentials: "same-origin" });
    if (!resp.ok) {
      throw new Error("Failed to load binary STL mesh (" + resp.status + ").");
    }
    const buf = await resp.arrayBuffer();
    const parsed = parseBinaryStl(buf, tintRgb);
    state.stlCache.set(url, parsed);
    return parsed;
  }

  // --- Resolve Tool Library Record & Cached 3D GPU Tool+Holder Mesh Per Segment ---
  function lookupCaseToolEntry(caseData, tNum, toolId) {
    if (toolId && state.toolLibraryById.has(toolId)) {
      return state.toolLibraryById.get(toolId);
    }
    const tbl = caseData && caseData.tools_table;
    if (!tbl) return null;
    if (Array.isArray(tbl)) {
      for (const item of tbl) {
        if (toolId && item.tool_id === toolId) return item;
        if (Number(item.tool_number || item.t_number) === Number(tNum)) return item;
      }
      return null;
    }
    return tbl[String(tNum)] || tbl["T" + String(tNum)] || (toolId ? tbl[toolId] : null) || null;
  }

  function resolveToolSpecForSegment(caseData, seg) {
    const fallback = (caseData && caseData.tool) || {};
    const tNum = Number((seg && seg.tool_number) || fallback.tool_number || 1);
    const toolId = (seg && seg.tool_id) || fallback.tool_id || "";
    const catRec = toolId && state.toolLibraryById.has(toolId) ? state.toolLibraryById.get(toolId) : null;
    const caseRec = lookupCaseToolEntry(caseData, tNum, toolId);
    const base = Object.assign({}, fallback, catRec || {}, caseRec || {});

    const rSeg = seg && seg.tool_radius_mm !== undefined ? Number(seg.tool_radius_mm) : null;
    const rCut = Math.max(
      0.2,
      Number(rSeg !== null && rSeg > 0 ? rSeg : base.cutter_radius_mm || (base.diameter_mm ? 0.5 * base.diameter_mm : 2.5))
    );
    return Object.assign({}, base, {
      tool_number: tNum,
      tool_id: base.tool_id || toolId || ("T" + tNum),
      entity_type: String((seg && seg.entity_type) || base.entity_type || "flat end mill"),
      cutter_radius_mm: rCut,
      diameter_mm: Number(base.diameter_mm || rCut * 2.0),
      corner_radius_mm: Number(
        seg && seg.corner_radius_mm !== undefined ? seg.corner_radius_mm : base.corner_radius_mm || 0.0
      ),
      tip_angle_deg: Number(
        seg && seg.tip_angle_deg !== undefined ? seg.tip_angle_deg : base.tip_angle_deg || 180.0
      ),
      flutes: Math.max(1, Number(base.flutes || 3)),
      flute_length_mm: Math.max(2.0, Number(base.flute_length_mm || rCut * 3.6)),
      shank_radius_mm: Math.max(0.25, Number(base.shank_radius_mm || (base.shank_diameter_mm ? 0.5 * base.shank_diameter_mm : rCut))),
      holder_bottom_mm: Math.max(6.0, Number(base.holder_bottom_mm || base.stickout_mm || rCut * 5.2)),
      holder_radius_mm: Math.max(rCut + 2.0, Number(base.holder_radius_mm || 16.0)),
      holder_height_mm: Math.max(18.0, Number(base.holder_height_mm || 30.0)),
      nominal_label: String(base.nominal_label || base.description || ""),
      unit_system: String(base.unit_system || "millimeters"),
      tool_material_id: String(base.tool_material_id || state.toolMaterialId || "carbide_coated"),
    });
  }

  function formatActiveToolLabel(spec) {
    if (!spec) return "T1";
    const tTag = "T" + Number(spec.tool_number || 1);
    if (spec.nominal_label) {
      const shortLbl = spec.nominal_label.length > 34 ? spec.nominal_label.slice(0, 33) + "…" : spec.nominal_label;
      return tTag + " · " + shortLbl;
    }
    const et = String(spec.entity_type || "flat end mill");
    return tTag + " · Ø" + Number(spec.diameter_mm || 5.0).toFixed(2) + "mm " + et;
  }

  function getOrUploadToolGpuMesh(spec) {
    if (!renderer || !spec) return null;
    const matId = String(state.toolMaterialId || spec.tool_material_id || "carbide_coated");
    const key = [
      spec.entity_type,
      Number(spec.cutter_radius_mm || 2.5).toFixed(3),
      Number(spec.corner_radius_mm || 0.0).toFixed(3),
      Number(spec.tip_angle_deg || 180.0).toFixed(1),
      Number(spec.flutes || 3),
      Number(spec.flute_length_mm || 12.0).toFixed(2),
      Number(spec.shank_radius_mm || 2.5).toFixed(3),
      Number(spec.holder_bottom_mm || 22.0).toFixed(2),
      Number(spec.holder_radius_mm || 16.0).toFixed(2),
      Number(spec.holder_height_mm || 30.0).toFixed(2),
      matId,
    ].join("|");

    if (renderer.toolMeshCache.has(key)) {
      return renderer.toolMeshCache.get(key);
    }
    const gl = renderer.gl;
    const built = buildToolAssemblyMesh(spec, matId);
    const entry = {
      pos: gl.createBuffer(),
      nrm: gl.createBuffer(),
      col: gl.createBuffer(),
      count: built.vertexCount,
    };
    uploadBuffer(gl, entry.pos, built.positions, false);
    uploadBuffer(gl, entry.nrm, built.normals, false);
    uploadBuffer(gl, entry.col, built.colors, false);
    renderer.toolMeshCache.set(key, entry);
    return entry;
  }

  function findSideBySideVariants(caseData) {
    const vars = (caseData && caseData.variants) || [];
    const naive = vars.find((v) => v.id === "nominal_cam") || vars[0];
    const rl =
      vars.find((v) => v.id === "shac_chipload") ||
      vars.find((v) => v.id === "shac_policy") ||
      naive;
    const direct =
      vars.find((v) => v.id === "chipload_ste") ||
      vars.find((v) => v.id === "ste_adam") ||
      naive;
    return [
      { variant: naive, title: "1. Naive Strategy (Nominal CAM)", best: false },
      { variant: rl, title: "2. RL Policy (SHAC Zero-Shot)", best: false },
      { variant: direct, title: "3. Direct DiffCAM (CWE Opt)", best: true },
    ];
  }

  function getActivePaneCount() {
    return state.layoutMode === "sbs" ? 3 : 1;
  }

  function formatRangeValue(val, mode) {
    if (mode === "chipload") return val.toFixed(4) + " mm";
    if (mode === "engagement") return val.toFixed(1) + "%D";
    if (mode === "mrr") return val.toFixed(1) + " mm³/s";
    return val.toFixed(1) + " mm/s";
  }

  // --- Update HUD Color Scale Legend (Tight Trajectory Range) ---
  function refreshHudLegend() {
    const titleEl = document.getElementById("legend-title");
    const barEl = document.getElementById("legend-bar");
    const minEl = document.getElementById("legend-min");
    const midEl = document.getElementById("legend-mid");
    const maxEl = document.getElementById("legend-max");
    const subEl = document.getElementById("legend-sub");

    if (state.stockMode === "live_deviation") {
      titleEl.textContent = "Stock-to-Model Deviation";
      barEl.className = "legend-gradient mode-deviation";
      minEl.textContent = "In-Tol (±0.15)";
      midEl.textContent = "Excess Stock";
      maxEl.textContent = "Gouge";
      subEl.textContent = "Live 2.5D heightfield deviation vs. CAD target";
      return;
    }

    const mode = state.pathColorMode;
    if (mode === "engaged_air") {
      titleEl.textContent = "Toolpath: Engaged Cut vs. Air-Cutting";
      barEl.className = "legend-gradient mode-engaged";
      minEl.textContent = "Engaged G1";
      midEl.textContent = "Air-Cut G1";
      maxEl.textContent = "Rapid G0";
      subEl.textContent = "Ribbon width: " + (state.pathWidthMm * 2.0).toFixed(2) + " mm · Drag: Orbit";
      return;
    }
    if (mode === "tool_number") {
      titleEl.textContent = "Multi-Tool TAMP Sequence (T1 → T" + Math.max(4, state.operations.length) + ")";
      barEl.className = "legend-gradient mode-tight";
      minEl.textContent = "T1 Rough/Face";
      midEl.textContent = "T2 Rest / T3 Spot";
      maxEl.textContent = "T4+ Drill/Finish";
      subEl.textContent = "Automated StandardToolLibrary + Persistent-IPW TAMP";
      return;
    }
    if (mode === "operation") {
      const nOps = Math.max(1, state.operations.length);
      const midOp = Math.max(1, Math.ceil(nOps * 0.5));
      titleEl.textContent = "Multi-Operation TAMP Sequence (Op 1 → Op " + nOps + ")";
      barEl.className = "legend-gradient mode-operation";
      minEl.textContent = "Op 1";
      midEl.textContent = "Op " + midOp;
      maxEl.textContent = "Op " + nOps;
      subEl.textContent = "Toggle Ops in top strip to isolate or overlay multiple operations";
      return;
    }

    const r = state.tightRange;
    const midVal = 0.5 * (r.min + r.max);
    barEl.className = "legend-gradient mode-tight";
    minEl.textContent = formatRangeValue(r.min, mode);
    midEl.textContent = r.constant ? "Uniform (" + formatRangeValue(midVal, mode) + ")" : formatRangeValue(midVal, mode);
    maxEl.textContent = formatRangeValue(r.max, mode);

    if (mode === "chipload") {
      const hOpt = state.baselinePhysics ? state.baselinePhysics.chip_thickness_mm : 0.035;
      titleEl.textContent = "Chip Load h_ex — Tight Fit (h_opt " + hOpt.toFixed(4) + " mm)";
      subEl.textContent = "Engaged G1 tightly mapped · Air G1: Slate · Rapid: Amber";
    } else if (mode === "engagement") {
      titleEl.textContent = "Radial Engagement a_e (%D) — Tight Fit";
      subEl.textContent = "Engaged G1 tightly mapped · Air G1: Slate · Rapid: Amber";
    } else if (mode === "mrr") {
      titleEl.textContent = "Instantaneous MRR (mm³/s) — Tight Fit";
      subEl.textContent = "Engaged G1 tightly mapped · Air G1: Slate · Rapid: Amber";
    } else {
      titleEl.textContent = "Programmed Cutting Feedrate — Tight Fit";
      subEl.textContent =
        r.constant
          ? "Fair Same-Speed Policy: all G1 cuts locked to " + formatRangeValue(r.min, mode)
          : "Full G1 trajectory feed variation tightly mapped";
    }
  }

  function getCaseZFlipSum(caseData) {
    if (!caseData) return 0.0;
    if (caseData.op2_z_flip_sum_mm !== undefined) {
      return Number(caseData.op2_z_flip_sum_mm);
    }
    return Number(caseData.stock_min_mm[2]) + Number(caseData.stock_max_mm[2]);
  }

  function isSegmentVisibleUnderOpFilter(seg) {
    if (state.opFilterMode !== "custom") return true;
    const key = resolveSegmentOpKey(seg);
    return state.selectedOpKeys.has(key);
  }

  function refreshOperationsForCurrentVariant(resetSelection) {
    const ops = deriveCaseOperations(state.currentCase, state.currentVariant);
    state.operations = ops;
    state.opByKey = new Map();
    state.opColorIndexMap = new Map();
    for (let i = 0; i < ops.length; i++) {
      state.opByKey.set(ops[i].key, ops[i]);
      state.opColorIndexMap.set(ops[i].key, ops[i].colorIdx);
    }

    if (resetSelection || state.opFilterMode !== "custom") {
      state.opFilterMode = "all";
      state.selectedOpKeys = new Set(ops.map((o) => o.key));
    } else {
      const kept = new Set();
      for (const op of ops) {
        if (state.selectedOpKeys.has(op.key)) kept.add(op.key);
      }
      if (kept.size === 0 || kept.size === ops.length) {
        state.opFilterMode = "all";
        state.selectedOpKeys = new Set(ops.map((o) => o.key));
      } else {
        state.selectedOpKeys = kept;
      }
    }
    renderOperationBar();
  }

  function applyOpSelectionSet(nextKeysSet, jumpToFirstSelected) {
    const allKeys = state.operations.map((o) => o.key);
    if (!nextKeysSet || nextKeysSet.size === 0 || nextKeysSet.size >= allKeys.length) {
      state.opFilterMode = "all";
      state.selectedOpKeys = new Set(allKeys);
    } else {
      state.opFilterMode = "custom";
      state.selectedOpKeys = new Set(nextKeysSet);
    }
    if (jumpToFirstSelected) {
      const firstOp = state.operations.find((o) => state.selectedOpKeys.has(o.key));
      if (firstOp) {
        state.playing = false;
        const playBtn = document.getElementById("btn-play");
        if (playBtn) playBtn.textContent = "Play";
        state.playTimeS = Number(firstOp.firstCutT0S !== undefined ? firstOp.firstCutT0S : firstOp.t0S);
      }
    }
    renderOperationBar();
    rebuildAllPaneToolpaths();
    updateScrubberAndDro();
    if (state.activeTab === "case") {
      renderInspectorTab();
    }
  }

  function scrubToOperation(opEntry, jumpToEnd) {
    if (!opEntry) return;
    if (state.opFilterMode === "custom" && !state.selectedOpKeys.has(opEntry.key)) {
      state.selectedOpKeys.add(opEntry.key);
      if (state.selectedOpKeys.size >= state.operations.length) {
        state.opFilterMode = "all";
      }
      rebuildAllPaneToolpaths();
    }
    state.playing = false;
    const playBtn = document.getElementById("btn-play");
    if (playBtn) playBtn.textContent = "Play";
    state.playTimeS = jumpToEnd
      ? Math.max(0.0, Number(opEntry.t1S) - 1e-4)
      : Number(opEntry.firstCutT0S !== undefined ? opEntry.firstCutT0S : opEntry.t0S);
    renderOperationBar();
    updateScrubberAndDro();
    if (state.activeTab === "case") {
      renderInspectorTab();
    }
  }

  function stepOperation(delta) {
    if (!state.operations || state.operations.length === 0) return;
    const curKey = state.activeOpKey;
    let curIdx = state.operations.findIndex((o) => o.key === curKey);
    if (curIdx < 0) curIdx = 0;
    const nextIdx = Math.max(0, Math.min(state.operations.length - 1, curIdx + delta));
    const targetOp = state.operations[nextIdx];
    scrubToOperation(targetOp, false);
  }

  function updateViseStatusPill(setupIdx) {
    const wh = state.workholding || resolveCaseWorkholding(state.currentCase);
    const sKey = Number(setupIdx || 1) >= 2 ? "2" : "1";
    const st = (wh && wh.setups && wh.setups[sKey]) || (wh && wh.setups && wh.setups["1"]) || null;
    const pill = document.getElementById("vise-status-pill");
    if (pill && st) {
      pill.textContent =
        st.short_label +
        " · " +
        Number(st.jaw_opening_mm || 0).toFixed(1) +
        "mm (" +
        String(st.clamping_axis || "Y") +
        ")";
      pill.title = String(st.label || wh.summary || "");
      pill.classList.toggle("vise-status-pill-soft", sKey === "2");
    }
    const droWh = document.getElementById("dro-workholding");
    if (droWh && st) {
      droWh.textContent =
        sKey === "2"
          ? "Kurt DX6 · G55 6061 Soft Jaws"
          : "Kurt DX6 · G54 Hard Jaws (" + Number(st.grip_z_mm || 2.0).toFixed(1) + "mm bite)";
    }
  }

  function renderOperationBar() {
    const strip = document.getElementById("op-pills-strip");
    const btnAll = document.getElementById("btn-ops-all");
    const btnG54 = document.getElementById("btn-ops-g54");
    const btnG55 = document.getElementById("btn-ops-g55");
    if (!strip) return;

    const ops = state.operations || [];
    const hasG55 = ops.some((o) => o.setupIndex >= 2);
    if (btnG54) btnG54.classList.toggle("hidden", !hasG55);
    if (btnG55) btnG55.classList.toggle("hidden", !hasG55);

    const isAll = state.opFilterMode !== "custom" || state.selectedOpKeys.size >= ops.length;
    const onlyG54 =
      hasG55 &&
      !isAll &&
      ops.every((o) => (o.setupIndex === 1) === state.selectedOpKeys.has(o.key));
    const onlyG55 =
      hasG55 &&
      !isAll &&
      ops.every((o) => (o.setupIndex >= 2) === state.selectedOpKeys.has(o.key));

    if (btnAll) btnAll.classList.toggle("btn-active", isAll);
    if (btnG54) btnG54.classList.toggle("btn-active", onlyG54);
    if (btnG55) btnG55.classList.toggle("btn-active", onlyG55);

    const pills = [];
    for (let i = 0; i < ops.length; i++) {
      const op = ops[i];
      const checked = isAll || state.selectedOpKeys.has(op.key);
      const isCurrent = state.activeOpKey === op.key;
      const clsParts = ["op-pill"];
      if (!checked) clsParts.push("op-pill-muted");
      if (isCurrent) clsParts.push("op-pill-current");
      if (op.setupIndex >= 2) clsParts.push("op-pill-g55");

      const chk = el("input", {
        type: "checkbox",
        className: "op-pill-chk",
        "aria-label": "Toggle visibility for Op " + op.opIndex + " " + op.shortKind,
        title: "Check/uncheck to overlay multiple operations simultaneously",
      });
      chk.checked = checked;
      chk.addEventListener("change", (ev) => {
        ev.stopPropagation();
        const nextSet = new Set(isAll ? ops.map((o) => o.key) : state.selectedOpKeys);
        if (chk.checked) {
          nextSet.add(op.key);
        } else {
          nextSet.delete(op.key);
        }
        applyOpSelectionSet(nextSet, false);
      });

      const jumpBtn = el(
        "button",
        {
          type: "button",
          className: "op-pill-jump",
          title:
            "Op " +
            op.opIndex +
            " (" +
            op.workOffset +
            " · T" +
            op.toolNumber +
            " · " +
            op.opKind +
            " · " +
            op.durationS.toFixed(1) +
            "s) — Click to scrub, Shift+Click to solo",
        },
        [
          el("span", { className: "op-swatch op-swatch-" + op.colorIdx }),
          el("span", {
            text:
              (hasG55 ? op.workOffset + " " : "") +
              "Op" +
              op.opIndex +
              " T" +
              op.toolNumber +
              " " +
              op.shortKind,
          }),
        ]
      );
      jumpBtn.addEventListener("click", (ev) => {
        if (ev.shiftKey) {
          applyOpSelectionSet(new Set([op.key]), true);
        } else {
          scrubToOperation(op, false);
        }
      });

      pills.push(el("div", { className: clsParts.join(" "), "data-op-key": op.key }, [chk, jumpBtn]));
    }
    strip.replaceChildren(...pills);
  }

  function syncOperationBarCurrentHighlight() {
    const strip = document.getElementById("op-pills-strip");
    if (!strip) return;
    for (const child of Array.from(strip.children)) {
      const k = child.getAttribute("data-op-key");
      child.classList.toggle("op-pill-current", Boolean(k && k === state.activeOpKey));
    }
  }

  function renderOperationTimelineSvg(playFrac) {
    const mount = document.getElementById("op-timeline-mount");
    if (!mount) return;
    const ops = state.operations || [];
    const totalT = Math.max(state.totalDurationS, 1e-6);
    const isAll = state.opFilterMode !== "custom" || state.selectedOpKeys.size >= ops.length;
    const children = [];

    let g55StartX = null;
    for (let i = 0; i < ops.length; i++) {
      const op = ops[i];
      const x0 = Math.max(0, Math.min(999, Math.round((op.t0S / totalT) * 1000)));
      const x1 = Math.max(x0 + 2, Math.min(1000, Math.round((op.t1S / totalT) * 1000)));
      const vis = isAll || state.selectedOpKeys.has(op.key);
      if (op.setupIndex >= 2 && g55StartX === null) {
        g55StartX = x0;
      }
      const rect = svgEl("rect", {
        x: String(x0),
        y: "0",
        width: String(Math.max(1, x1 - x0 - 1)),
        height: "13",
        fill: op.colorHex,
        "fill-opacity": vis ? (state.activeOpKey === op.key ? "1.0" : "0.78") : "0.18",
      });
      rect.appendChild(
        svgEl("title", null, [
          "Op " +
            op.opIndex +
            " (" +
            op.workOffset +
            " T" +
            op.toolNumber +
            " " +
            op.shortKind +
            "): " +
            op.t0S.toFixed(1) +
            "s – " +
            op.t1S.toFixed(1) +
            "s",
        ])
      );
      rect.addEventListener("click", (ev) => {
        ev.stopPropagation();
        scrubToOperation(op, false);
      });
      children.push(rect);
    }

    if (g55StartX !== null) {
      children.push(
        svgEl("rect", {
          x: String(Math.max(0, g55StartX - 1)),
          y: "0",
          width: "3",
          height: "13",
          fill: "#ffd43b",
        })
      );
    }

    const playX = Math.max(0, Math.min(997, Math.round(playFrac * 1000) - 1));
    children.push(
      svgEl("rect", {
        x: String(playX),
        y: "0",
        width: "3",
        height: "13",
        fill: "#ffffff",
      })
    );

    const svg = svgEl(
      "svg",
      { viewBox: "0 0 1000 13", preserveAspectRatio: "none", class: "op-timeline-svg" },
      children
    );
    mount.replaceChildren(svg);
  }

  // --- Upload 3D Thick Ribbon Buffers for All Active Panes ---
  function rebuildAllPaneToolpaths() {
    if (!renderer || !state.currentCase) return;
    const gl = renderer.gl;
    const nPanes = getActivePaneCount();
    const activePlans = [];
    for (let p = 0; p < nPanes; p++) {
      const slot = renderer.panes[p];
      if (slot.variant && slot.variant.plan) {
        activePlans.push(slot.variant.plan);
      }
    }
    state.tightRange = computeTightColorRange(activePlans, state.pathColorMode);

    const cutRadius = state.pathWidthMm;
    const rapidRadius = Math.max(0.08, state.pathWidthMm * 0.58);
    const zFlipSum = getCaseZFlipSum(state.currentCase);
    const segFilter = state.opFilterMode === "custom" ? isSegmentVisibleUnderOpFilter : null;

    for (let p = 0; p < nPanes; p++) {
      const slot = renderer.panes[p];
      if (!slot.variant || !slot.variant.plan) continue;
      const segs = slot.variant.plan.segments || [];

      const cutGeom = buildThickPathTubeGeometry(
        segs,
        false,
        cutRadius,
        (s) => cutSegmentColorRgb(s, state.pathColorMode, state.tightRange, state.opColorIndexMap),
        zFlipSum,
        segFilter
      );
      uploadBuffer(gl, slot.cutPos, cutGeom.positions, false);
      uploadBuffer(gl, slot.cutNrm, cutGeom.normals, false);
      uploadBuffer(gl, slot.cutCol, cutGeom.colors, false);
      slot.cutSegIndices = cutGeom.segIndices;
      slot.cutVertsPerSeg = cutGeom.vertsPerSeg;
      slot.cutTotalVerts = cutGeom.totalVerts;

      const rapGeom = buildThickPathTubeGeometry(
        segs,
        true,
        rapidRadius,
        () => [0.96, 0.65, 0.14],
        zFlipSum,
        segFilter
      );
      uploadBuffer(gl, slot.rapidPos, rapGeom.positions, false);
      uploadBuffer(gl, slot.rapidNrm, rapGeom.normals, false);
      uploadBuffer(gl, slot.rapidCol, rapGeom.colors, false);
      slot.rapidSegIndices = rapGeom.segIndices;
      slot.rapidVertsPerSeg = rapGeom.vertsPerSeg;
      slot.rapidTotalVerts = rapGeom.totalVerts;
    }

    refreshHudLegend();
  }

  function renderPhysicsReadoutStrip() {
    const strip = document.getElementById("physics-readout-strip");
    if (!strip || !state.baselinePhysics) return;
    const r = state.baselinePhysics;
    const isClamped = r.constraint_feed_scale < 0.995;

    strip.replaceChildren(
      el("span", {
        className: "phys-pill",
        text:
          "Ø" +
          r.diameter_mm.toFixed(1) +
          "mm " +
          r.flutes +
          "FL · Stickout " +
          r.stickout_mm.toFixed(0) +
          "mm · DOC " +
          r.doc_mm.toFixed(1) +
          " / WOC " +
          r.woc_mm.toFixed(2) +
          "mm",
      }),
      el("span", {
        className: "phys-pill phys-pill-cyan",
        text:
          Math.round(r.spindle_rpm) +
          " RPM (" +
          Math.round(r.surface_speed_sfm) +
          " SFM) · Base " +
          r.feed_mm_s.toFixed(1) +
          " mm/s (" +
          Math.round(r.feed_mm_min) +
          " mm/min)",
      }),
      el("span", {
        className: "phys-pill phys-pill-cyan",
        text: "Target h_opt " + r.chip_thickness_mm.toFixed(4) + " mm/tooth",
      }),
      el("span", {
        className: isClamped ? "phys-pill phys-pill-warn" : "phys-pill",
        text:
          "Defl " +
          r.deflection_utilization_pct.toFixed(0) +
          "% · Power " +
          r.power_utilization_pct.toFixed(0) +
          "% · " +
          r.active_limit,
      })
    );
  }

  function syncVariantButtonsLabels() {
    if (!state.currentCase) return;
    const btnWrap = document.getElementById("variant-buttons");
    if (!btnWrap) return;
    for (const child of Array.from(btnWrap.children)) {
      const vid = child.getAttribute("data-vid");
      const v = state.currentCase.variants.find((item) => item.id === vid);
      if (v) {
        child.textContent = v.label + " (" + v.metrics.cycle_time_s.toFixed(1) + "s)";
        if (state.currentVariant && v.id === state.currentVariant.id) {
          child.classList.add("btn-active");
        } else {
          child.classList.remove("btn-active");
        }
      }
    }
  }

  function refreshActivePaneAssignments(preservePlayFrac) {
    if (!state.currentCase || !renderer) return;
    state.gcodeProgramCache = null;
    state.lastHighlightedLine = -1;
    const prevFrac =
      preservePlayFrac && state.totalDurationS > 1e-6
        ? Math.max(0.0, Math.min(1.0, state.playTimeS / state.totalDurationS))
        : null;

    if (state.layoutMode === "sbs") {
      const sbsSpec = findSideBySideVariants(state.currentCase);
      let maxDur = 0.0;
      for (let i = 0; i < 3; i++) {
        const slot = renderer.panes[i];
        const v = sbsSpec[i].variant;
        slot.variant = v;
        slot.roleTitle = sbsSpec[i].title;
        slot.isBest = sbsSpec[i].best;
        if (!slot.dexelSim) {
          slot.dexelSim = new DexelStockSim(state.currentCase, DEXEL_PITCH_MM);
        }
        slot.dexelSim.setSegments((v && v.plan && v.plan.segments) || []);
        maxDur = Math.max(maxDur, Number((v && v.plan && v.plan.total_duration_s) || 0.0));
        loadStaticStlIntoSlot(slot, v);
      }
      state.currentVariant = sbsSpec[2].variant || state.currentVariant;
      state.totalDurationS = maxDur;
      document.getElementById("hud-dro").classList.add("hidden");
      document.getElementById("sbs-overlay").classList.remove("hidden");
      document.getElementById("single-variant-group").classList.add("hidden");
    } else {
      const v = state.currentVariant || state.currentCase.variants[0];
      const slot = renderer.panes[0];
      slot.variant = v;
      slot.roleTitle = v ? v.label : "";
      slot.isBest = false;
      if (!slot.dexelSim) {
        slot.dexelSim = new DexelStockSim(state.currentCase, DEXEL_PITCH_MM);
      }
      slot.dexelSim.setSegments((v && v.plan && v.plan.segments) || []);
      state.totalDurationS = Number((v && v.plan && v.plan.total_duration_s) || 0.0);
      loadStaticStlIntoSlot(slot, v);
      document.getElementById("hud-dro").classList.remove("hidden");
      document.getElementById("sbs-overlay").classList.add("hidden");
      document.getElementById("single-variant-group").classList.remove("hidden");
    }

    if (prevFrac !== null) {
      state.playTimeS = prevFrac * state.totalDurationS;
    } else if (state.firstBoot) {
      state.firstBoot = false;
      state.playTimeS = 0.0;
      state.playing = true;
    }
    document.getElementById("btn-play").textContent = state.playing ? "Pause" : "Play";

    refreshOperationsForCurrentVariant(false);
    rebuildAllPaneToolpaths();
    renderInspectorTab();
    updateScrubberAndDro();
    renderKpiStrip();
  }

  async function loadStaticStlIntoSlot(slot, variant) {
    if (!variant || !variant.stock_stl_url || !renderer) return;
    const mySeq = state.loadSeq;
    try {
      const stockMesh = await fetchParsedStl(variant.stock_stl_url, [0.74, 0.80, 0.87]);
      if (mySeq !== state.loadSeq || slot.variant !== variant) return;
      const gl = renderer.gl;
      uploadBuffer(gl, slot.staticStockPos, stockMesh.positions, false);
      uploadBuffer(gl, slot.staticStockNrm, stockMesh.normals, false);
      uploadBuffer(gl, slot.staticStockCol, stockMesh.colors, false);
      slot.staticStockCount = stockMesh.vertexCount;
    } catch (_err) {
      // Non-fatal: live dexel stock remains primary
    }
  }

  function reapplyPhysicsPolicyToCase(preserveTimeFrac) {
    if (!state.currentCase) return;
    const cutter = buildCutterSpecFromCase(state.currentCase);
    const nomDoc = Number(state.currentCase.nominal_doc_mm || 3.0);
    const nomWoc = Number(state.currentCase.nominal_woc_mm || 1.75);

    const baseRes = computeFeedsAndSpeeds(
      {
        stock_material_id: state.stockMaterialId,
        tool_material_id: state.toolMaterialId,
        cutter,
        doc_mm: nomDoc,
        woc_mm: nomWoc,
        aggressiveness: state.aggressiveness,
      },
      state.catalog
    );
    state.baselinePhysics = baseRes;
    renderPhysicsReadoutStrip();

    const policySettings = {
      speedPolicy: state.speedPolicy,
      aggressiveness: state.aggressiveness,
      stockMaterialId: state.stockMaterialId,
      toolMaterialId: state.toolMaterialId,
    };

    for (const v of state.currentCase.variants) {
      evaluateVariantUnderPolicy(v, state.currentCase, policySettings, baseRes, state.catalog);
    }

    const naiveVar = state.currentCase.variants.find((v) => v.id === "nominal_cam");
    const naiveTime = naiveVar ? Number(naiveVar.metrics.cycle_time_s) : 0.0;
    for (const v of state.currentCase.variants) {
      v.metrics.nominal_cycle_time_s = naiveTime;
      v.metrics.cycle_time_reduction_pct =
        naiveTime > 1e-6
          ? Number(((100.0 * (naiveTime - v.metrics.cycle_time_s)) / naiveTime).toFixed(2))
          : 0.0;
    }

    state.gcodeProgramCache = null;
    syncVariantButtonsLabels();
    refreshActivePaneAssignments(preserveTimeFrac);
  }

  // --- Client + Server Multi-Dialect G-Code Synthesizer & Line Mapper ---
  const DIALECT_LABELS = {
    haas_ngc: { label: "Haas NGC (.nc)", ext: ".nc", smooth: "G187 P2 E0.05", cancelSmooth: "G187" },
    rs274: { label: "LinuxCNC RS-274 (.ngc)", ext: ".ngc", smooth: "G64 P0.05", cancelSmooth: "" },
    fanuc: { label: "FANUC 31i-B (.nc)", ext: ".nc", smooth: "G05.1 Q1 R5", cancelSmooth: "G05.1 Q0" },
    robodrill: { label: "FANUC Robodrill (.nc)", ext: ".nc", smooth: "G05.1 Q1 R5", cancelSmooth: "G05.1 Q0" },
    makino: { label: "Makino Pro6 (.nc)", ext: ".nc", smooth: "M252 (SGI.4)", cancelSmooth: "M250" },
    mazak: { label: "Mazak SmoothG (.eia)", ext: ".eia", smooth: "G61.1 P2", cancelSmooth: "G64" },
  };

  function sanitizeGcodeComment(raw) {
    return String(raw || "")
      .replace(/[()%\r\n]/g, " ")
      .replace(/[^\x20-\x7e]/g, "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 64);
  }

  function buildGcodeProgramForActiveVariant(dialectId) {
    const dial = DIALECT_LABELS[dialectId] ? dialectId : "haas_ngc";
    const caseData = state.currentCase;
    const variant = state.currentVariant;
    if (!caseData || !variant || !variant.plan) {
      return { dialect: dial, lines: ["(NO PROGRAM LOADED)"], segToLine: [], lineToSeg: new Map() };
    }

    const cacheKey = [
      caseData.case_id,
      variant.id,
      dial,
      state.speedPolicy,
      state.aggressiveness.toFixed(2),
      state.stockMaterialId,
      state.toolMaterialId,
    ].join("|");
    if (state.gcodeProgramCache && state.gcodeProgramCache.key === cacheKey) {
      return state.gcodeProgramCache;
    }

    const segs = variant.plan.segments || [];
    // Prefer server-posted shop G-code when embedded and running default 100% Optimal Chipload on original alloy
    const embedded =
      variant.gcode_by_dialect &&
      variant.gcode_by_dialect[dial] &&
      state.speedPolicy === "optimal_chipload" &&
      Math.abs(state.aggressiveness - 1.0) < 1e-3 &&
      (!caseData.stock_material_id || caseData.stock_material_id === state.stockMaterialId)
        ? variant.gcode_by_dialect[dial]
        : null;

    const embeddedLines =
      embedded && Array.isArray(embedded.lines) && embedded.lines.length > 0
        ? embedded.lines
        : embedded && typeof embedded.text === "string" && embedded.text.length > 0
          ? embedded.text.split("\n")
          : null;

    if (embedded && embeddedLines && embeddedLines.length > 0) {
      const segToLine = Array.isArray(embedded.viewer_segment_to_line) && embedded.viewer_segment_to_line.length === segs.length
        ? embedded.viewer_segment_to_line.slice()
        : Array.isArray(embedded.segment_to_line)
          ? embedded.segment_to_line.slice()
          : [];
      const lineToSeg = new Map();
      for (let i = 0; i < segToLine.length; i++) {
        const ln = Number(segToLine[i] || 0);
        if (ln > 0 && !lineToSeg.has(ln)) {
          lineToSeg.set(ln, i);
        }
      }
      const progObj = {
        key: cacheKey,
        dialect: dial,
        lines: embeddedLines,
        text: embedded.text || embeddedLines.join("\n"),
        segToLine,
        lineToSeg,
      };
      state.gcodeProgramCache = progObj;
      return progObj;
    }

    // Live client-side 6-dialect G-code formatter (reflects live Aggressiveness / Material / Speed Policy!)
    const info = DIALECT_LABELS[dial];
    const lines = [];
    const segToLine = new Array(segs.length).fill(1);
    const lineToSeg = new Map();

    if (dial !== "mazak" && dial !== "rs274") {
      lines.push("%");
    }
    const oWord = dial === "haas_ngc" ? "O01001" : "O1001";
    const safeTitle = sanitizeGcodeComment(caseData.case_id + " - " + variant.label);
    if (dial !== "rs274") {
      lines.push(oWord + " (" + safeTitle + ")");
    } else {
      lines.push("(" + safeTitle + ")");
    }
    lines.push(
      "(MAT: " +
        sanitizeGcodeComment(state.stockMaterialId.toUpperCase()) +
        " | POLICY: " +
        (state.speedPolicy === "matched" ? "SAME-SPEED" : "OPT-CHIPLOAD") +
        " " +
        Math.round(state.aggressiveness * 100) +
        "%)"
    );

    // Emit Tool Crib Table header & collect cutting tools for Renishaw OTS calibration
    const seenTools = new Set();
    const otsToolSpecs = [];
    for (let i = 0; i < segs.length; i++) {
      const tNum = Number(segs[i].tool_number || 1);
      if (!seenTools.has(tNum)) {
        seenTools.add(tNum);
        const tSpec = resolveToolSpecForSegment(caseData, segs[i]);
        otsToolSpecs.push({ tNum, tSpec });
        lines.push(
          "(T" +
            tNum +
            " D=" +
            Number(tSpec.diameter_mm || 5.0).toFixed(3) +
            "MM " +
            sanitizeGcodeComment(tSpec.nominal_label || tSpec.entity_type) +
            ")"
        );
      }
    }

    if (dial === "rs274") {
      lines.push("G17 G21 G40 G49 G80 G90 G91.1 G94");
    } else {
      lines.push("G00 G17 G21 G40 G49 G80 G90 G94");
    }

    // Emit Haas WIPS-R / Renishaw OTS Automatic Tool Setting Preamble for cutting tools
    const cuttingForOts = otsToolSpecs.filter(
      (item) => String(item.tSpec.entity_type || "").toLowerCase() !== "probe" && item.tNum < 90
    );
    if (cuttingForOts.length > 0) {
      lines.push("(--- HAAS WIPS-R : RENISHAW OTS AUTO TOOL SETTING ---)");
      for (const item of cuttingForOts) {
        const dMm = Number(item.tSpec.diameter_mm || 5.0);
        const stkMm = Number(item.tSpec.holder_bottom_mm || 28.0);
        const otsMode = classifyOtsModeForTool(item.tSpec.entity_type, dMm, true);
        const bArg = otsMode.mode === "ots_length_and_diameter" ? "B3." : "B1.";
        lines.push("T" + item.tNum + " M06");
        if (dial === "haas_ngc") {
          lines.push(
            "G65 P9995 A0. " +
              bArg +
              " C2. T" +
              item.tNum +
              ". E" +
              stkMm.toFixed(2) +
              " D" +
              dMm.toFixed(3) +
              " (OTS SET H" +
              item.tNum +
              ")"
          );
        } else if (dial === "rs274") {
          lines.push("G38.2 Z-25.000 F120.0 (OTS TOUCH TOOL T" + item.tNum + ")");
          lines.push("G10 L1 P" + item.tNum + " Z#5063 R" + (0.5 * dMm).toFixed(4));
        } else {
          lines.push(
            "G65 P9857 " + bArg + " T" + item.tNum + ". D" + dMm.toFixed(3) + " (OTS SET H" + item.tNum + ")"
          );
        }
      }
      lines.push(dial === "fanuc" || dial === "makino" ? "G28 G91 Z0.0" : "G90 G53 G00 Z0.0");
      if (dial === "fanuc" || dial === "makino") lines.push("G90");
    }

    // If case does not already start with an explicit T99 probe segment, emit Step-1 OMP40-2 WCS probe!
    const sminFb = (caseData.stock_min_mm || [-25, -20, -10]).map(Number);
    const smaxFb = (caseData.stock_max_mm || [25, 20, 0]).map(Number);
    const sxFb = Math.max(4.0, smaxFb[0] - sminFb[0]).toFixed(3);
    const syFb = Math.max(4.0, smaxFb[1] - sminFb[1]).toFixed(3);
    const firstIsProbe =
      segs.length > 0 &&
      (Number(segs[0].tool_number || 1) >= 90 || String(segs[0].entity_type || "").toLowerCase() === "probe");
    if (!firstIsProbe) {
      lines.push("(--- STEP 1 : RENISHAW OMP40-2 WCS PROBE G54 ---)");
      lines.push("T99 M06");
      lines.push("G54");
      lines.push("G90 G00 X0.000 Y0.000");
      lines.push("G43 H99 Z25.000");
      if (dial === "haas_ngc") {
        lines.push("G65 P9832 (WIPS-R OMP40-2 OPTICAL PROBE ON)");
        lines.push("G65 P9810 Z10.000 F1500.0");
        lines.push("G65 P9811 Z0.000 Q5.000 S1. (SET G54 TOP Z0 DATUM)");
        lines.push("G65 P9812 X" + sxFb + " Z-3.500 Q5.000 R8.000 S1. (SET G54 X WEB CENTER)");
        lines.push("G65 P9812 Y" + syFb + " Z-3.500 Q5.000 R8.000 S1. (SET G54 Y WEB CENTER)");
        lines.push("G65 P9995 W54. A20. X" + sxFb + " Y" + syFb + " Z-3.500");
        lines.push("G65 P9833 (WIPS-R OMP40-2 OPTICAL PROBE OFF)");
      } else if (dial === "rs274") {
        lines.push("G38.2 Z-5.000 F150.0 (PROBE G54 TOP Z0)");
        lines.push("G10 L20 P1 Z0.000");
        lines.push("G00 Z15.000");
      } else {
        lines.push("G65 P9832 (OMP40-2 PROBE ON)");
        lines.push("G65 P9810 Z10.000 F1500.0");
        lines.push("G65 P9811 Z0.000 Q5.000 S1.");
        lines.push("G65 P9812 X" + sxFb + " Z-3.500 Q5.000 R8.000 S1.");
        lines.push("G65 P9812 Y" + syFb + " Z-3.500 Q5.000 R8.000 S1.");
        lines.push("G65 P9833 (OMP40-2 PROBE OFF)");
      }
      lines.push(dial === "fanuc" || dial === "makino" ? "G28 G91 Z0.0" : "G90 G53 G00 Z0.0");
      if (dial === "fanuc" || dial === "makino") lines.push("G90");
    }

    let curTool = -1;
    let curSetup = 1;
    let curWorkOffset = "";
    let lastModal = "";
    let lastFeedStr = "";

    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      const nextSetup = Number(s.setup_index || 1);
      const nextWcs = String(s.work_offset || (nextSetup === 2 ? "G55" : "G54"));
      const nextTool = Number(s.tool_number || 1);

      if (nextSetup !== curSetup && curTool > 0) {
        if (info.cancelSmooth) lines.push(info.cancelSmooth);
        lines.push("M09");
        lines.push("M05");
        lines.push(dial === "fanuc" || dial === "makino" ? "G28 G91 Z0.0" : "G90 G53 G00 Z0.0");
        if (dial === "fanuc" || dial === "makino") lines.push("G90");
        lines.push("M00 (OP2 FLIP PART 180 DEG IN " + sanitizeGcodeComment(nextWcs) + " SOFT JAWS)");
        curSetup = nextSetup;
        curTool = -1; // Force tool offset re-activation on Op2
      } else {
        curSetup = nextSetup;
      }

      if (nextTool !== curTool || nextWcs !== curWorkOffset) {
        if (curTool > 0 && nextTool !== curTool) {
          if (info.cancelSmooth) lines.push(info.cancelSmooth);
          lines.push("M09");
          lines.push("M05");
          lines.push(dial === "fanuc" || dial === "makino" ? "G28 G91 Z0.0" : "G90 G53 G00 Z0.0");
          if (dial === "fanuc" || dial === "makino") lines.push("G90");
        }
        const tSpec = resolveToolSpecForSegment(caseData, s);
        const isProbe = String(tSpec.entity_type || "").toLowerCase() === "probe" || nextTool >= 90;
        if (nextTool !== curTool) {
          lines.push("(TOOL T" + nextTool + " - " + sanitizeGcodeComment(tSpec.nominal_label || tSpec.entity_type) + ")");
          lines.push("T" + nextTool + " M06");
        }
        lines.push(nextWcs);
        if (isProbe) {
          lines.push("G43 H" + nextTool + " Z25.000");
          if (dial === "haas_ngc") {
            lines.push("G65 P9832 (ACTIVATE SPINDLE PROBE)");
            lines.push("G65 P9995 W" + (nextSetup === 2 ? "55." : "54.") + " A20. (WIPS AUTO WCS PROBE)");
          } else {
            lines.push("(SPINDLE PROBE CALIBRATE " + nextWcs + ")");
          }
        } else {
          const rpm = Math.max(400, Math.round(Number(s.spindle_rpm || (state.baselinePhysics && state.baselinePhysics.spindle_rpm) || 8000)));
          lines.push("S" + rpm + " M03");
          lines.push("G43 H" + nextTool + " Z" + Number(s.start[2] || 15.0).toFixed(3));
          lines.push("M08");
          if (info.smooth) lines.push(info.smooth);
        }
        curTool = nextTool;
        curWorkOffset = nextWcs;
        lastModal = "";
        lastFeedStr = "";
      }

      if (String(s.class || "") === "tool_change") {
        const ln = lines.length;
        segToLine[i] = ln;
        if (!lineToSeg.has(ln)) lineToSeg.set(ln, i);
        continue;
      }

      const isRapid = s.kind === "rapid";
      const gCode = isRapid ? "G00" : "G01";
      const parts = [];
      if (gCode !== lastModal) {
        parts.push(gCode);
        lastModal = gCode;
      }
      parts.push("X" + Number(s.end[0]).toFixed(3));
      parts.push("Y" + Number(s.end[1]).toFixed(3));
      parts.push("Z" + Number(s.end[2]).toFixed(3));
      if (!isRapid) {
        const fMin = Math.max(5.0, Number(s.feed_mm_min || Number(s.feed_mm_s || 10.0) * 60.0));
        const fStr = fMin.toFixed(1);
        if (fStr !== lastFeedStr) {
          parts.push("F" + fStr);
          lastFeedStr = fStr;
        }
      }
      lines.push(parts.join(" "));
      const lineNum = lines.length; // 1-based!
      segToLine[i] = lineNum;
      if (!lineToSeg.has(lineNum)) {
        lineToSeg.set(lineNum, i);
      }
    }

    if (info.cancelSmooth) lines.push(info.cancelSmooth);
    lines.push("M09");
    lines.push("M05");
    lines.push(dial === "fanuc" || dial === "makino" ? "G28 G91 Z0.0" : "G90 G53 G00 Z0.0");
    if (dial === "fanuc" || dial === "makino") lines.push("G90");
    lines.push("M30");
    if (dial !== "mazak" && dial !== "rs274") {
      lines.push("%");
    }

    const progObj = {
      key: cacheKey,
      dialect: dial,
      lines,
      text: lines.join("\n"),
      segToLine,
      lineToSeg,
    };
    state.gcodeProgramCache = progObj;
    return progObj;
  }

  function resolveGcodeLineForSeg(segIndex, activeSeg) {
    const prog = buildGcodeProgramForActiveVariant(state.gcodeDialect);
    if (segIndex >= 0 && segIndex < prog.segToLine.length && prog.segToLine[segIndex] > 0) {
      return prog.segToLine[segIndex];
    }
    if (activeSeg && activeSeg.gcode_line) {
      return Number(activeSeg.gcode_line);
    }
    return 1;
  }

  function highlightActiveGcodeLine(lineNum) {
    if (state.activeTab !== "gcode") return;
    if (lineNum === state.lastHighlightedLine) return;
    const viewport = document.getElementById("gcode-viewport");
    if (!viewport) return;

    if (state.lastHighlightedLine > 0) {
      const prevEl = viewport.querySelector('[data-ln="' + state.lastHighlightedLine + '"]');
      if (prevEl) prevEl.classList.remove("gcode-line-active");
    }
    state.lastHighlightedLine = lineNum;
    const nextEl = viewport.querySelector('[data-ln="' + lineNum + '"]');
    if (nextEl) {
      nextEl.classList.add("gcode-line-active");
      if (state.gcodeAutoFollow) {
        const top = nextEl.offsetTop - viewport.offsetTop;
        const viewH = viewport.clientHeight;
        if (top < viewport.scrollTop + 24 || top > viewport.scrollTop + viewH - 44) {
          viewport.scrollTop = Math.max(0, top - Math.floor(viewH * 0.38));
        }
      }
    }
    const statusLn = document.getElementById("gcode-active-ln");
    if (statusLn) {
      statusLn.textContent = "Active Block: Line " + lineNum;
    }
  }

  // --- Evaluate Pose, Active Tool Assembly, & Live Dexel Carve for a Single Pane Slot at time t ---
  function stepPaneSlotAtTime(slot, t) {
    if (!slot.variant || !slot.variant.plan) return;
    const plan = slot.variant.plan;
    const segs = plan.segments || [];
    const varDur = Number(plan.total_duration_s || 0.0);
    const clampedT = Math.max(0.0, Math.min(t, varDur));
    slot.finished = varDur > 1e-6 && t >= varDur - 1e-4;

    let activeSegIndex = -1;
    let alpha = 0.0;
    let pos = plan.initial_pos ? plan.initial_pos.slice() : [0, 0, 6];
    let activeSeg = null;

    if (clampedT <= 1e-7 || segs.length === 0) {
      activeSegIndex = -1;
      alpha = 0.0;
    } else {
      for (let i = 0; i < segs.length; i++) {
        const s = segs[i];
        if (clampedT <= s.t1_s + 1e-7 || i === segs.length - 1) {
          activeSegIndex = i;
          activeSeg = s;
          const span = Math.max(s.t1_s - s.t0_s, 1e-9);
          alpha = Math.max(0.0, Math.min(1.0, (clampedT - s.t0_s) / span));
          pos = [
            s.start[0] + (s.end[0] - s.start[0]) * alpha,
            s.start[1] + (s.end[1] - s.start[1]) * alpha,
            s.start[2] + (s.end[2] - s.start[2]) * alpha,
          ];
          break;
        }
      }
    }

    slot.toolPos = pos;
    slot.activeSegIndex = activeSegIndex;
    slot.activeSegAlpha = alpha;
    slot.activeSeg = activeSeg;

    const refSegForTool = activeSeg || (segs.length > 0 ? segs[0] : null);
    const toolSpec = resolveToolSpecForSegment(state.currentCase, refSegForTool);
    slot.activeToolGpu = getOrUploadToolGpuMesh(toolSpec);
    slot.activeToolLabel = formatActiveToolLabel(toolSpec);
    slot.activeSetupIndex = activeSeg ? Number(activeSeg.setup_index || 1) : 1;
    slot.activeWorkOffset =
      activeSeg && activeSeg.work_offset
        ? String(activeSeg.work_offset)
        : slot.activeSetupIndex === 2
          ? "G55"
          : "G54";
    slot.activeOpKey = refSegForTool ? resolveSegmentOpKey(refSegForTool) : "";
    const opMeta = state.opByKey.get(slot.activeOpKey);
    slot.activeOpLabel = opMeta
      ? "Op " + opMeta.opIndex + " / " + Math.max(1, state.operations.length) + " · " + opMeta.shortKind
      : "Op 1 / " + Math.max(1, state.operations.length);
    slot.activeGcodeLine = resolveGcodeLineForSeg(activeSegIndex, activeSeg);

    if (slot.dexelSim && state.stockMode !== "static_stl") {
      const mesh = slot.dexelSim.carveAndTriangulate(
        activeSegIndex,
        alpha,
        state.stockMode,
        state.stockPalette
      );
      if (mesh) {
        const gl = renderer.gl;
        uploadBuffer(gl, slot.liveStockPos, mesh.positions, true);
        uploadBuffer(gl, slot.liveStockNrm, mesh.normals, true);
        uploadBuffer(gl, slot.liveStockCol, mesh.colors, true);
        slot.liveStockCount = mesh.vertexCount;
      }
    }

    if (activeSegIndex >= 0 && activeSeg) {
      const zFlipSum = getCaseZFlipSum(state.currentCase);
      const activeTube = buildThickPathTubeGeometry(
        [activeSeg],
        activeSeg.kind === "rapid",
        state.pathWidthMm * 1.45,
        () => [1.0, 1.0, 1.0],
        zFlipSum
      );
      const gl = renderer.gl;
      uploadBuffer(gl, slot.activePos, activeTube.positions, true);
      uploadBuffer(gl, slot.activeNrm, activeTube.normals, true);
      uploadBuffer(gl, slot.activeCol, activeTube.colors, true);
      slot.activeVerts = activeTube.totalVerts;
    } else {
      slot.activeVerts = 0;
    }
  }

  function renderSideBySideOverlayCards() {
    const overlay = document.getElementById("sbs-overlay");
    if (!overlay || state.layoutMode !== "sbs") return;

    const naiveDur =
      renderer.panes[0].variant && renderer.panes[0].variant.plan
        ? Number(renderer.panes[0].variant.plan.total_duration_s || 0.0)
        : 0.0;

    const cols = [];
    for (let i = 0; i < 3; i++) {
      const slot = renderer.panes[i];
      const v = slot.variant;
      if (!v) continue;
      const p = v.plan;
      const eng = p.engagement_summary || {};
      const dur = Number(p.total_duration_s || 0.0);
      const gainPct = naiveDur > 1e-6 ? ((naiveDur - dur) / naiveDur) * 100.0 : 0.0;

      let badgeText = "READY";
      let badgeCls = "state-badge state-engaged";
      if (slot.finished) {
        badgeText = "DONE ✓";
        badgeCls = "state-badge state-done";
      } else if (slot.activeSeg) {
        if (slot.activeSeg.kind === "rapid") {
          badgeText = "RAPID G0";
          badgeCls = "state-badge state-rapid";
        } else if (slot.activeSeg.engaged) {
          badgeText = "ENGAGED";
          badgeCls = "state-badge state-engaged";
        } else {
          badgeText = "AIR G1";
          badgeCls = "state-badge state-air";
        }
      }

      const deltaStr =
        i === 0
          ? "Baseline"
          : (gainPct >= 0 ? "-" : "+") + Math.abs(gainPct).toFixed(1) + "% vs Naive";
      const curFeed = slot.activeSeg ? Number(slot.activeSeg.feed_mm_s || 0.0) : 0.0;
      const curHex = slot.activeSeg && slot.activeSeg.engaged ? Number(slot.activeSeg.hex_mm || 0.0) : 0.0;

      cols.push(
        el("div", { className: "sbs-pane-col" }, [
          el("div", { className: slot.isBest ? "sbs-card sbs-card-best" : "sbs-card" }, [
            el("div", { className: "sbs-card-head" }, [
              el("span", { className: "sbs-card-title", text: slot.roleTitle }),
              el("span", { className: badgeCls, text: badgeText }),
            ]),
            el("div", { className: "sbs-metrics-grid" }, [
              el("span", { className: "sbs-kv-cyan", text: "Cycle: " + dur.toFixed(2) + "s" }),
              el("span", {
                className: gainPct >= 0.05 ? "sbs-kv-good" : "sbs-kv-muted",
                text: deltaStr,
              }),
              el(
                "span",
                { className: "sbs-kv-muted" },
                [
                  slot.activeWorkOffset +
                    " " +
                    (state.opByKey.has(slot.activeOpKey) ? "Op" + state.opByKey.get(slot.activeOpKey).opIndex + " · " : "· ") +
                    slot.activeToolLabel.slice(0, 18),
                ]
              ),
              el(
                "span",
                { className: "sbs-kv-muted" },
                ["Opt h_ex: " + Number(eng.optimal_hex_pct || 0).toFixed(0) + "%"]
              ),
              el("span", { className: "sbs-kv-muted", text: "Feed: " + curFeed.toFixed(1) + " mm/s" }),
              el("span", { className: "sbs-kv-muted", text: "h_ex: " + curHex.toFixed(4) + " mm" }),
            ]),
          ]),
        ])
      );
    }
    overlay.replaceChildren(...cols);
  }

  // --- Render Stacked Engaged vs. Air-Cutting vs. Rapid SVG Bar ---
  function renderEngagementBar(playFrac) {
    const mount = document.getElementById("eng-svg-mount");
    const legendRow = document.getElementById("eng-legend-row");
    const volEl = document.getElementById("eng-vol-summary");
    const cardTitle = document.getElementById("eng-card-title");
    if (!state.currentVariant) return;

    cardTitle.textContent =
      state.layoutMode === "sbs"
        ? "Direct DiffCAM — Engaged vs. Air Breakdown"
        : "Time Engaged vs. Air-Cutting Breakdown";

    const eng = state.currentVariant.plan.engagement_summary || {};
    const ePct = Number(eng.engaged_pct || 0.0);
    const aPct = Number(eng.air_cut_pct || 0.0);
    const rPct = Math.max(0.0, 100.0 - ePct - aPct);

    const wEng = Math.round(ePct * 10);
    const wAir = Math.round(aPct * 10);
    const wRap = Math.max(0, 1000 - wEng - wAir);
    const playX = Math.max(0, Math.min(996, Math.round(playFrac * 1000) - 2));

    const svg = svgEl("svg", { viewBox: "0 0 1000 14", preserveAspectRatio: "none", class: "eng-bar-svg" }, [
      svgEl("rect", { x: "0", y: "0", width: String(wEng), height: "14", class: "svg-seg-engaged" }),
      svgEl("rect", { x: String(wEng), y: "0", width: String(wAir), height: "14", class: "svg-seg-air" }),
      svgEl("rect", { x: String(wEng + wAir), y: "0", width: String(wRap), height: "14", class: "svg-seg-rapid" }),
      svgEl("rect", { x: String(playX), y: "0", width: "4", height: "14", class: "svg-playhead" }),
    ]);
    mount.replaceChildren(svg);

    volEl.textContent = Number(eng.removed_volume_mm3 || 0).toFixed(1) + " mm³ total cut";
    legendRow.replaceChildren(
      el("span", { className: "dot-pill" }, [
        el("span", { className: "dot dot-engaged" }),
        "Engaged: " + Number(eng.engaged_time_s || 0).toFixed(2) + "s (" + ePct.toFixed(1) + "%)",
      ]),
      el("span", { className: "dot-pill" }, [
        el("span", { className: "dot dot-air" }),
        "Air G1: " + Number(eng.air_cut_time_s || 0).toFixed(2) + "s (" + aPct.toFixed(1) + "%)",
      ]),
      el("span", { className: "dot-pill" }, [
        el("span", { className: "dot dot-rapid" }),
        "Rapid G0: " + Number(eng.rapid_time_s || 0).toFixed(2) + "s (" + rPct.toFixed(1) + "%)",
      ])
    );
  }

  // --- Update Timeline Scrubber, Step Active Pane(s), & Refresh DRO / G-Code Line ---
  function updateScrubberAndDro() {
    if (!renderer || !state.currentCase) return;
    const t = Math.max(0.0, Math.min(state.playTimeS, state.totalDurationS));
    const playFrac = state.totalDurationS > 1e-6 ? t / state.totalDurationS : 1.0;

    const slider = document.getElementById("time-slider");
    slider.value = String(Math.round(playFrac * 1000));
    document.getElementById("time-readout").textContent =
      t.toFixed(2) + " / " + state.totalDurationS.toFixed(2) + " s";

    const nPanes = getActivePaneCount();
    for (let i = 0; i < nPanes; i++) {
      stepPaneSlotAtTime(renderer.panes[i], t);
    }

    const refSlot = state.layoutMode === "sbs" ? renderer.panes[2] : renderer.panes[0];
    if (refSlot) {
      state.activeOpKey = refSlot.activeOpKey || "";
      highlightActiveGcodeLine(refSlot.activeGcodeLine);
      updateViseStatusPill(refSlot.activeSetupIndex);
    }
    syncOperationBarCurrentHighlight();
    renderOperationTimelineSvg(playFrac);
    renderEngagementBar(playFrac);

    if (state.layoutMode === "sbs") {
      renderSideBySideOverlayCards();
      return;
    }

    const slot = renderer.panes[0];
    if (!slot.variant) return;
    const plan = slot.variant.plan;
    const segs = plan.segments || [];
    const activeSeg = slot.activeSeg;
    const activeSegIndex = slot.activeSegIndex;
    const alpha = slot.activeSegAlpha;
    const pos = slot.toolPos;

    const engSum = plan.engagement_summary || {};
    const totalVol = Number(engSum.removed_volume_mm3 || 0.0);
    let liveVol = 0.0;
    if (activeSegIndex >= 0 && activeSeg) {
      const prevCum = activeSegIndex > 0 ? Number(segs[activeSegIndex - 1].cum_removed_mm3 || 0.0) : 0.0;
      liveVol = prevCum + alpha * Number(activeSeg.removed_mm3 || 0.0);
    }

    const badge = document.getElementById("dro-state-badge");
    if (!activeSeg) {
      badge.textContent = "BILLET READY";
      badge.className = "state-badge state-engaged";
    } else if (slot.finished) {
      badge.textContent = "FINISHED ✓";
      badge.className = "state-badge state-done";
    } else if (activeSeg.kind === "rapid") {
      badge.textContent = "RAPID G0";
      badge.className = "state-badge state-rapid";
    } else if (activeSeg.engaged) {
      badge.textContent = "ENGAGED CUT";
      badge.className = "state-badge state-engaged";
    } else {
      badge.textContent = "AIR-CUT G1";
      badge.className = "state-badge state-air";
    }

    const feedS = activeSeg ? Number(activeSeg.feed_mm_s || 0.0) : 0.0;
    const feedMin = activeSeg ? Number(activeSeg.feed_mm_min || feedS * 60.0) : 0.0;
    const aeMm = activeSeg ? Number(activeSeg.ae_mm || 0.0) : 0.0;
    const aePct = activeSeg ? Number(activeSeg.ae_pct || 0.0) : 0.0;
    const apMm = activeSeg ? Number(activeSeg.ap_mm || 0.0) : 0.0;
    const hexMm = activeSeg ? Number(activeSeg.hex_mm || 0.0) : 0.0;

    document.getElementById("dro-x").textContent = pos[0].toFixed(3) + " mm";
    document.getElementById("dro-y").textContent = pos[1].toFixed(3) + " mm";
    document.getElementById("dro-z").textContent = pos[2].toFixed(3) + " mm";
    const droToolEl = document.getElementById("dro-tool");
    if (droToolEl) droToolEl.textContent = slot.activeToolLabel;
    const droOpEl = document.getElementById("dro-op");
    if (droOpEl) droOpEl.textContent = slot.activeOpLabel;
    const droSetupEl = document.getElementById("dro-setup-line");
    if (droSetupEl) {
      droSetupEl.textContent =
        slot.activeWorkOffset +
        " Op" +
        slot.activeSetupIndex +
        " · Line " +
        slot.activeGcodeLine;
    }
    document.getElementById("dro-class").textContent = activeSeg ? String(activeSeg.class || activeSeg.kind) : "Start (Uncut)";
    document.getElementById("dro-feed").textContent =
      feedS.toFixed(1) + " mm/s (" + Math.round(feedMin) + " mm/min)";
    document.getElementById("dro-ae").textContent =
      "a_e " + aeMm.toFixed(2) + "mm (" + aePct.toFixed(0) + "%D) · a_p " + apMm.toFixed(1) + "mm";
    document.getElementById("dro-hex").textContent = hexMm.toFixed(4) + " mm/tooth";
    document.getElementById("dro-vol").textContent =
      liveVol.toFixed(0) + " / " + totalVol.toFixed(0) + " mm³";
    document.getElementById("dro-seg").textContent =
      (activeSegIndex + 1) + " / " + segs.length;
  }

  // --- Select Variant Within Current Case ---
  function selectVariant(variantId, preservePlayhead) {
    if (!state.currentCase) return;
    const variant =
      state.currentCase.variants.find((v) => v.id === variantId) || state.currentCase.variants[0];
    if (!variant) return;

    state.currentVariant = variant;
    state.gcodeProgramCache = null;
    syncVariantButtonsLabels();

    if (state.currentCase.rustcnc_download_url) {
      document.getElementById("dl-rustcnc").setAttribute("href", state.currentCase.rustcnc_download_url);
    }
    if (variant.camotics_project_url) {
      document.getElementById("dl-camotics").setAttribute("href", variant.camotics_project_url);
    }
    const dlNc = document.getElementById("dl-nc");
    if (variant.haas_nc_url) {
      dlNc.setAttribute("href", variant.haas_nc_url);
      dlNc.textContent = "Haas NGC (.nc)";
    } else if (variant.camotics_nc_url) {
      dlNc.setAttribute("href", variant.camotics_nc_url);
      dlNc.textContent = "RS-274 (.nc)";
    }

    refreshActivePaneAssignments(preservePlayhead);
  }

  // --- Load Benchmark or Uploaded CAD Case ---
  async function loadCase(caseJsonUrl, preferredVariantId) {
    const mySeq = ++state.loadSeq;
    try {
      clearError();
      const caseData = await fetchJson(caseJsonUrl);
      if (mySeq !== state.loadSeq) return;
      state.currentCase = caseData;
      state.currentCaseUrl = caseJsonUrl;
      state.gcodeProgramCache = null;

      for (let i = 0; i < 3; i++) {
        renderer.panes[i].dexelSim = new DexelStockSim(caseData, DEXEL_PITCH_MM);
      }

      const gl = renderer.gl;
      state.workholding = resolveCaseWorkholding(caseData);
      state.opFilterMode = "all";
      state.selectedOpKeys = new Set();

      const viseMesh1 = buildKurtDx6ViseMesh(caseData, 1);
      uploadBuffer(gl, renderer.shared.vise1Pos, viseMesh1.positions, false);
      uploadBuffer(gl, renderer.shared.vise1Nrm, viseMesh1.normals, false);
      uploadBuffer(gl, renderer.shared.vise1Col, viseMesh1.colors, false);
      state.vise1Count = viseMesh1.vertexCount;

      const viseMesh2 = buildKurtDx6ViseMesh(caseData, 2);
      uploadBuffer(gl, renderer.shared.vise2Pos, viseMesh2.positions, false);
      uploadBuffer(gl, renderer.shared.vise2Nrm, viseMesh2.normals, false);
      uploadBuffer(gl, renderer.shared.vise2Col, viseMesh2.colors, false);
      state.vise2Count = viseMesh2.vertexCount;

      const guides = buildSceneGuideLines(caseData.stock_min_mm, caseData.stock_max_mm, state.workholding);
      uploadBuffer(gl, renderer.shared.guidePos, guides.positions, false);
      uploadBuffer(gl, renderer.shared.guideCol, guides.colors, false);
      state.guideGridCount = guides.gridCount;
      state.guideBilletOffset = guides.billetOffset;
      state.guideBilletCount = guides.billetCount;

      const toolMesh = buildToolAssemblyMesh(caseData.tool || {}, state.toolMaterialId);
      uploadBuffer(gl, renderer.shared.toolPos, toolMesh.positions, false);
      uploadBuffer(gl, renderer.shared.toolNrm, toolMesh.normals, false);
      uploadBuffer(gl, renderer.shared.toolCol, toolMesh.colors, false);
      state.toolCount = toolMesh.vertexCount;

      const targetMesh = await fetchParsedStl(caseData.target_stl_url, [0.18, 0.85, 0.64]);
      if (mySeq !== state.loadSeq) return;
      uploadBuffer(gl, renderer.shared.targetPos, targetMesh.positions, false);
      uploadBuffer(gl, renderer.shared.targetNrm, targetMesh.normals, false);
      state.targetCount = targetMesh.vertexCount;

      // For arbitrary STEP/STL or multi-tool CAD parts, rasterize actual target STL into DexelStockSim!
      if (caseData.is_arbitrary_cad || caseData.tools_table || caseData.operations_table) {
        for (let i = 0; i < 3; i++) {
          if (renderer.panes[i].dexelSim) {
            renderer.panes[i].dexelSim.setTargetFromMesh(targetMesh);
          }
        }
      }

      const cx = 0.5 * (caseData.stock_min_mm[0] + caseData.stock_max_mm[0]);
      const cy = 0.5 * (caseData.stock_min_mm[1] + caseData.stock_max_mm[1]);
      const cz = 0.5 * (caseData.stock_min_mm[2] + caseData.stock_max_mm[2]);
      state.cam.target = [cx, cy, cz];
      state.cam.distance = defaultCamDistance(caseData);

      // Show Re-Plan w/ Crib button when interactive server is active and part has CAD source
      const replanBtn = document.getElementById("btn-replan-current");
      if (replanBtn) {
        const canReplan = Boolean(
          state.serverInteractive &&
            (caseData.can_replan || caseData.source_cad_id || String(caseData.case_id || "").startsWith("titan-"))
        );
        if (canReplan) replanBtn.classList.remove("hidden");
        else replanBtn.classList.add("hidden");
      }

      // Auto-switch Path Color to Tool Number when loading a multi-tool TAMP / STEP part if user was on default
      if (
        caseData.tools_table &&
        Object.keys(caseData.tools_table).length > 1 &&
        state.firstBoot === false &&
        caseData.default_path_color === "tool_number"
      ) {
        state.pathColorMode = "tool_number";
        const pcSel = document.getElementById("path-color-select");
        if (pcSel) pcSel.value = "tool_number";
      }

      const btnWrap = document.getElementById("variant-buttons");
      btnWrap.replaceChildren();
      for (const v of caseData.variants) {
        const b = el("button", {
          type: "button",
          className: "btn",
          "data-vid": v.id,
          text: v.label,
        });
        b.addEventListener("click", () => {
          state.playing = false;
          state.playTimeS = Number((v.plan && v.plan.total_duration_s) || 0.0);
          selectVariant(v.id, false);
        });
        btnWrap.appendChild(b);
      }

      const defaultVid =
        caseData.variants.some((v) => v.id === "chipload_ste") ? "chipload_ste" : "ste_adam";
      const targetVid =
        preferredVariantId ||
        (state.currentVariant && state.currentVariant.id) ||
        defaultVid;
      state.currentVariant =
        caseData.variants.find((v) => v.id === targetVid) || caseData.variants[0];

      reapplyPhysicsPolicyToCase(false);
      selectVariant(state.currentVariant.id, true);
    } catch (err) {
      showError(err.message || "Unable to load benchmark part.");
    }
  }

  // --- Render 6-Card KPI Strip ---
  function renderKpiStrip() {
    const strip = document.getElementById("kpi-strip");
    strip.replaceChildren();
    if (!state.currentVariant) return;
    const m = state.currentVariant.metrics;
    const p = state.currentVariant.plan;
    const eng = p.engagement_summary || {};
    const policyTag = state.speedPolicy === "matched" ? "Same Speed" : "Opt Chipload";
    const gainSign =
      Math.abs(m.cycle_time_reduction_pct) >= 0.05
        ? (m.cycle_time_reduction_pct >= 0 ? "-" : "+") +
          Math.abs(m.cycle_time_reduction_pct).toFixed(1) +
          "% vs Naive (" +
          policyTag +
          ")"
        : "Naive Baseline (" + policyTag + ")";
    const safetyOk = Boolean(m.posted_geometry_pass && m.posted_safety_pass);
    const band = eng.optimal_hex_band_mm || [0.02, 0.045];

    const cards = [
      {
        label: "Cycle Time",
        value: m.cycle_time_s.toFixed(2) + " s",
        sub: gainSign,
        cls: "kpi-cyan",
      },
      {
        label: "Engaged Cut (G1)",
        value: Number(eng.engaged_time_s || 0).toFixed(2) + " s",
        sub: Number(eng.engaged_pct || 0).toFixed(1) + "% · " + Number(eng.engaged_dist_mm || 0).toFixed(0) + " mm",
        cls: "kpi-good",
      },
      {
        label: "Air-Cutting (G1)",
        value: Number(eng.air_cut_time_s || 0).toFixed(2) + " s",
        sub: Number(eng.air_cut_pct || 0).toFixed(1) + "% · " + Number(eng.air_cut_dist_mm || 0).toFixed(0) + " mm",
        cls: "kpi-rose",
      },
      {
        label: "Mean Chip h_ex",
        value: Number(eng.mean_hex_mm || 0).toFixed(4) + " mm",
        sub:
          Number(eng.optimal_hex_pct || 0).toFixed(0) +
          "% in [" +
          Number(band[0]).toFixed(3) +
          ", " +
          Number(band[1]).toFixed(3) +
          "]",
        cls: Number(eng.optimal_hex_pct || 0) >= 65 ? "kpi-good" : "kpi-amber",
      },
      {
        label: "Mean Feed / Peak MRR",
        value: Number(p.mean_cut_feed_mm_s || 0).toFixed(1) + " mm/s",
        sub: "Peak " + m.peak_mrr_mm3_s.toFixed(0) + " mm³/s · a_e " + Number(eng.mean_ae_pct || 0).toFixed(0) + "%D",
        cls: "",
      },
      {
        label: "XY Reshape / Safety",
        value: "ΔXY " + Number(p.max_xy_shift_mm || 0).toFixed(2) + " mm",
        sub: safetyOk ? "0.125mm PASS (0 gouge)" : "CHECK",
        cls: safetyOk ? "kpi-good" : "kpi-amber",
      },
    ];

    for (const c of cards) {
      strip.appendChild(
        el("div", { className: "kpi-card" }, [
          el("div", { className: "kpi-label", text: c.label }),
          el("div", { className: ("kpi-value " + c.cls).trim(), text: c.value }),
          el("div", { className: "kpi-sub", text: c.sub }),
        ])
      );
    }
  }

  function buildTable(headers, rows, highlightPred) {
    const thead = el("thead", null, [
      el("tr", null, headers.map((h) => el("th", { text: h }))),
    ]);
    const tbody = el("tbody");
    for (const r of rows) {
      const tr = el("tr", { className: highlightPred && highlightPred(r) ? "row-highlight" : "" });
      for (const cell of r.cells) {
        if (cell && typeof cell === "object" && cell.nodeType === 1) {
          const td = el("td");
          td.appendChild(cell);
          tr.appendChild(td);
        } else {
          tr.appendChild(el("td", { text: String(cell) }));
        }
      }
      tbody.appendChild(tr);
    }
    return el("table", { className: "data-table" }, [thead, tbody]);
  }

  // --- Syntax-Colored Safe DOM Tokenizer for G-Code Lines (Zero innerHTML) ---
  function buildSyntaxColoredCodeSpan(lineStr) {
    const codeWrap = el("span", { className: "gcode-code" });
    const trimmed = String(lineStr || "");
    if (!trimmed) return codeWrap;

    if (trimmed.startsWith("(") || trimmed.startsWith(";") || trimmed === "%") {
      codeWrap.appendChild(el("span", { className: "gc-tok-comment", text: trimmed }));
      return codeWrap;
    }

    const commentIdx = trimmed.indexOf("(");
    const mainPart = commentIdx >= 0 ? trimmed.slice(0, commentIdx) : trimmed;
    const commentPart = commentIdx >= 0 ? trimmed.slice(commentIdx) : "";

    const tokens = mainPart.split(/(\s+)/);
    for (const tok of tokens) {
      if (!tok) continue;
      if (/^\s+$/.test(tok)) {
        codeWrap.appendChild(document.createTextNode(tok));
        continue;
      }
      const c0 = tok.charAt(0).toUpperCase();
      let cls = "gc-tok-coord";
      if (c0 === "G") cls = "gc-tok-g";
      else if (c0 === "M") cls = "gc-tok-m";
      else if (c0 === "T" || c0 === "H" || c0 === "O") cls = "gc-tok-t";
      else if (c0 === "F" || c0 === "S") cls = "gc-tok-f";
      codeWrap.appendChild(el("span", { className: cls, text: tok }));
    }
    if (commentPart) {
      codeWrap.appendChild(el("span", { className: "gc-tok-comment", text: commentPart }));
    }
    return codeWrap;
  }

  function renderGcodeTab(container) {
    const prog = buildGcodeProgramForActiveVariant(state.gcodeDialect);
    const dialMeta = DIALECT_LABELS[state.gcodeDialect] || DIALECT_LABELS.haas_ngc;

    const dialSelect = el("select", { className: "select-compact", "aria-label": "G-Code controller dialect" });
    for (const [did, meta] of Object.entries(DIALECT_LABELS)) {
      dialSelect.appendChild(el("option", { value: did, text: meta.label }));
    }
    dialSelect.value = state.gcodeDialect;
    dialSelect.addEventListener("change", (e) => {
      state.gcodeDialect = String(e.target.value || "haas_ngc");
      state.lastHighlightedLine = -1;
      renderInspectorTab();
      const refSlot = state.layoutMode === "sbs" ? renderer.panes[2] : renderer.panes[0];
      if (refSlot) {
        refSlot.activeGcodeLine = resolveGcodeLineForSeg(refSlot.activeSegIndex, refSlot.activeSeg);
        highlightActiveGcodeLine(refSlot.activeGcodeLine);
      }
    });

    const followChk = el("input", { type: "checkbox" });
    followChk.checked = state.gcodeAutoFollow;
    followChk.addEventListener("change", () => {
      state.gcodeAutoFollow = followChk.checked;
    });

    const copyBtn = el("button", { type: "button", className: "btn btn-tiny", text: "Copy" });
    copyBtn.addEventListener("click", async () => {
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(prog.text);
          copyBtn.textContent = "Copied ✓";
          setTimeout(() => {
            copyBtn.textContent = "Copy";
          }, 1400);
        }
      } catch (_err) {
        copyBtn.textContent = "Copy N/A";
      }
    });

    const dlBtn = el("button", { type: "button", className: "btn btn-tiny btn-accent", text: "Download " + dialMeta.ext });
    dlBtn.addEventListener("click", () => {
      const blob = new Blob([prog.text + "\n"], { type: "text/plain;charset=utf-8" });
      const objUrl = URL.createObjectURL(blob);
      const a = el("a", {
        href: objUrl,
        download: (state.currentCase ? state.currentCase.case_id : "program") + "_" + state.gcodeDialect + dialMeta.ext,
      });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(objUrl), 2000);
    });

    const toolbar = el("div", { className: "gcode-toolbar" }, [
      dialSelect,
      el("div", { className: "segmented-group" }, [
        el("label", { className: "toggle-pill" }, [followChk, el("span", { text: "Auto-Scroll" })]),
        copyBtn,
        dlBtn,
      ]),
    ]);

    const statusBar = el("div", { className: "gcode-status-bar" }, [
      el("span", { id: "gcode-active-ln", text: "Active Block: Line 1" }),
      el("span", { text: prog.lines.length + " lines · Click any line to scrub 3D tool" }),
    ]);

    const viewport = el("div", { id: "gcode-viewport", className: "gcode-viewport" });
    const segs = (state.currentVariant && state.currentVariant.plan && state.currentVariant.plan.segments) || [];

    const maxRenderLines = Math.min(prog.lines.length, 4500);
    for (let idx = 0; idx < maxRenderLines; idx++) {
      const ln = idx + 1;
      const row = el("div", { className: "gcode-line", "data-ln": String(ln) }, [
        el("span", { className: "gcode-ln", text: String(ln) }),
        buildSyntaxColoredCodeSpan(prog.lines[idx]),
      ]);
      row.addEventListener("click", () => {
        // Find matching or nearest preceding segment for this G-code line
        let targetSegIdx = -1;
        if (prog.lineToSeg.has(ln)) {
          targetSegIdx = prog.lineToSeg.get(ln);
        } else {
          for (let probeLn = ln; probeLn >= 1; probeLn--) {
            if (prog.lineToSeg.has(probeLn)) {
              targetSegIdx = prog.lineToSeg.get(probeLn);
              break;
            }
          }
        }
        state.playing = false;
        document.getElementById("btn-play").textContent = "Play";
        if (targetSegIdx >= 0 && targetSegIdx < segs.length) {
          state.playTimeS = Number(segs[targetSegIdx].t1_s || 0.0);
        } else if (ln <= 5) {
          state.playTimeS = 0.0;
        }
        updateScrubberAndDro();
      });
      viewport.appendChild(row);
    }

    const panel = el("div", { className: "gcode-panel" }, [toolbar, statusBar, viewport]);
    container.appendChild(panel);

    state.lastHighlightedLine = -1;
    const refSlot = state.layoutMode === "sbs" ? renderer.panes[2] : renderer.panes[0];
    if (refSlot) {
      highlightActiveGcodeLine(refSlot.activeGcodeLine);
    }
  }

  // --- Right-Side Tool Crib Tab (Metric / US Library + Available Tool Selection) ---
  function classifyToolCategory(t) {
    const et = String(t.entity_type || "").toLowerCase();
    if (et === "probe") return "probe";
    if (et === "ball end mill" || et === "lollipop mill") return "ball";
    if (et === "flat end mill" || et === "bull nose end mill" || et === "tapered mill") return "endmill";
    if (et === "drill" || et === "spot drill" || et === "center drill" || et === "reamer") return "drill";
    if (et === "chamfer mill" || et === "counter sink") return "chamfer";
    if (et === "thread mill" || et === "tap right hand") return "thread";
    return "specialty";
  }

  function updateModalCribBadge() {
    const badge = document.getElementById("modal-crib-count");
    if (badge) {
      badge.textContent = String(state.selectedToolIds.size);
    }
  }

  function applyStarterCribPreset(cribId) {
    const crib = state.starterCribs && state.starterCribs[cribId];
    if (crib && Array.isArray(crib.tool_ids)) {
      if (crib.unit_system) {
        state.unitSystem = crib.unit_system;
        const machUnit = document.getElementById("mach-unit-system");
        if (machUnit) machUnit.value = crib.unit_system;
      }
      state.selectedToolIds = new Set(crib.tool_ids);
      updateModalCribBadge();
      if (state.activeTab === "tools") renderInspectorTab();
    }
  }

  function renderToolCribTab(container) {
    const all3AxisTools = state.toolLibrary.filter((t) => t.is_3axis_mill_compatible !== false);

    // 1. Unit Library Segmented Selector (US Inch | Metric mm | Both)
    const unitBtns = [
      { id: "inches", label: "US Library (inch)" },
      { id: "millimeters", label: "Metric Library (mm)" },
      { id: "both", label: "Combined" },
    ].map((u) => {
      const b = el("button", {
        type: "button",
        className: state.unitSystem === u.id ? "btn btn-tiny btn-active" : "btn btn-tiny",
        text: u.label,
      });
      b.addEventListener("click", () => {
        state.unitSystem = u.id;
        const machUnit = document.getElementById("mach-unit-system");
        if (machUnit) machUnit.value = u.id;
        // Auto-apply matching starter crib if current selection has 0 tools in the newly selected unit
        const hasMatchingChecked = all3AxisTools.some(
          (t) => state.selectedToolIds.has(t.tool_id) && (u.id === "both" || t.unit_system === u.id)
        );
        if (!hasMatchingChecked) {
          const fallbackCrib = u.id === "millimeters" ? "metric_haas_10t" : "us_haas_10t";
          if (state.starterCribs[fallbackCrib]) {
            state.selectedToolIds = new Set(state.starterCribs[fallbackCrib].tool_ids);
            updateModalCribBadge();
          }
        }
        renderInspectorTab();
      });
      return b;
    });

    const topRow = el("div", { className: "crib-top-row" }, [
      el("div", { className: "segmented-group" }, unitBtns),
    ]);

    // 2. Starter Crib Presets + Select All / Clear
    const presetRow = el("div", { className: "crib-preset-row" });
    const presetDefs = [
      { id: "us_haas_10t", short: "US Shop 12T" },
      { id: "us_minimal_4t", short: "US Min 4T" },
      { id: "metric_haas_10t", short: "Metric Shop 13T" },
      { id: "metric_minimal_4t", short: "Metric Min 4T" },
    ];
    for (const p of presetDefs) {
      if (!state.starterCribs[p.id]) continue;
      const b = el("button", { type: "button", className: "btn btn-tiny", text: p.short });
      b.addEventListener("click", () => applyStarterCribPreset(p.id));
      presetRow.appendChild(b);
    }

    const btnAllUnit = el("button", { type: "button", className: "btn btn-tiny", text: "All in Unit" });
    btnAllUnit.addEventListener("click", () => {
      for (const t of all3AxisTools) {
        if (state.unitSystem === "both" || t.unit_system === state.unitSystem) {
          state.selectedToolIds.add(t.tool_id);
        }
      }
      updateModalCribBadge();
      renderInspectorTab();
    });
    const btnClear = el("button", { type: "button", className: "btn btn-tiny", text: "Clear" });
    btnClear.addEventListener("click", () => {
      state.selectedToolIds.clear();
      updateModalCribBadge();
      renderInspectorTab();
    });
    presetRow.appendChild(btnAllUnit);
    presetRow.appendChild(btnClear);

    // 3. Search + Category Filter
    const searchInput = el("input", {
      type: "text",
      className: "text-input",
      placeholder: 'Filter tools (e.g. 1/4", #7, ball, chamfer, M6)...',
      value: state.cribSearch,
    });
    const catSelect = el("select", { className: "select-compact", "aria-label": "Filter tool category" }, [
      el("option", { value: "all", text: "All Categories" }),
      el("option", { value: "checked", text: "Checked Only" }),
      el("option", { value: "endmill", text: "Flat & Bull Mills" }),
      el("option", { value: "ball", text: "Ball Mills (3D)" }),
      el("option", { value: "drill", text: "Drills & Spot" }),
      el("option", { value: "chamfer", text: "Chamfer & C-Sink" }),
      el("option", { value: "thread", text: "Thread & Taps" }),
      el("option", { value: "specialty", text: "Face & Specialty" }),
      el("option", { value: "probe", text: "Spindle Probes" }),
    ]);
    catSelect.value = state.cribCategory;

    const listContainer = el("div", { className: "crib-list" });
    const countSpan = el("span", { className: "accent-mono" });

    function populateFilteredList() {
      const q = state.cribSearch.trim().toLowerCase();
      const filtered = all3AxisTools.filter((t) => {
        if (state.unitSystem !== "both" && t.unit_system !== state.unitSystem) return false;
        if (state.cribCategory === "checked" && !state.selectedToolIds.has(t.tool_id)) return false;
        if (state.cribCategory !== "all" && state.cribCategory !== "checked") {
          if (classifyToolCategory(t) !== state.cribCategory) return false;
        }
        if (q) {
          const hay = (t.nominal_label + " " + t.entity_type + " " + t.tool_id + " " + t.diameter_mm).toLowerCase();
          if (hay.indexOf(q) < 0) return false;
        }
        return true;
      });

      const checkedInUnit = all3AxisTools.filter(
        (t) => state.selectedToolIds.has(t.tool_id) && (state.unitSystem === "both" || t.unit_system === state.unitSystem)
      ).length;
      countSpan.textContent = checkedInUnit + " available tools selected (" + filtered.length + " shown)";

      listContainer.replaceChildren();
      const cap = Math.min(filtered.length, 260);
      for (let i = 0; i < cap; i++) {
        const t = filtered[i];
        const isChecked = state.selectedToolIds.has(t.tool_id);
        const chk = el("input", { type: "checkbox" });
        chk.checked = isChecked;

        const isInch = t.unit_system === "inches";
        const diamText = isInch
          ? t.diameter_in.toFixed(4) + '" (' + t.diameter_mm.toFixed(2) + "mm)"
          : "Ø" + t.diameter_mm.toFixed(2) + " mm";

        const row = el(
          "label",
          { className: isChecked ? "crib-item crib-item-checked" : "crib-item" },
          [
            el("div", { className: "crib-item-left" }, [
              chk,
              el("div", null, [
                el("div", { className: "crib-item-title", text: t.nominal_label }),
                el("div", {
                  className: "crib-item-sub",
                  text:
                    diamText +
                    " · " +
                    t.flutes +
                    "FL · LOC " +
                    t.flute_length_mm.toFixed(1) +
                    "mm · " +
                    (t.holder_taper || "BT30"),
                }),
              ]),
            ]),
            el("div", { className: "crib-badges" }, [
              el("span", {
                className: isInch ? "crib-badge crib-badge-us" : "crib-badge crib-badge-mm",
                text: isInch ? "US" : "MM",
              }),
            ]),
          ]
        );

        chk.addEventListener("change", () => {
          if (chk.checked) {
            state.selectedToolIds.add(t.tool_id);
            row.classList.add("crib-item-checked");
          } else {
            state.selectedToolIds.delete(t.tool_id);
            row.classList.remove("crib-item-checked");
          }
          updateModalCribBadge();
          const newChecked = all3AxisTools.filter(
            (item) =>
              state.selectedToolIds.has(item.tool_id) &&
              (state.unitSystem === "both" || item.unit_system === state.unitSystem)
          ).length;
          countSpan.textContent = newChecked + " available tools selected (" + filtered.length + " shown)";
        });

        listContainer.appendChild(row);
      }
    }

    searchInput.addEventListener("input", () => {
      state.cribSearch = searchInput.value;
      populateFilteredList();
    });
    catSelect.addEventListener("change", () => {
      state.cribCategory = catSelect.value;
      populateFilteredList();
    });

    const replanBtn = el("button", {
      type: "button",
      className: "btn btn-tiny btn-primary",
      text: state.serverInteractive ? "Re-Plan Part w/ Crib" : "+ Machine CAD w/ Crib",
    });
    replanBtn.addEventListener("click", () => {
      const canReplanCurrent = Boolean(
        state.serverInteractive &&
          state.currentCase &&
          (state.currentCase.can_replan ||
            state.currentCase.source_cad_id ||
            String(state.currentCase.case_id || "").startsWith("titan-"))
      );
      if (canReplanCurrent) {
        triggerReplanCurrentPart();
      } else {
        openUploadModal();
      }
    });

    const summaryStrip = el("div", { className: "crib-summary-strip" }, [countSpan, replanBtn]);
    const filterRow = el("div", { className: "crib-filter-row" }, [searchInput, catSelect]);

    populateFilteredList();

    container.appendChild(
      el("div", { className: "crib-panel" }, [topRow, presetRow, filterRow, summaryStrip, listContainer])
    );
  }

  function renderInspectorTab() {
    const container = document.getElementById("tab-content");
    container.replaceChildren();
    if (!state.currentCase || !state.manifest) return;

    if (state.activeTab === "gcode") {
      renderGcodeTab(container);
      return;
    }

    if (state.activeTab === "tools") {
      renderToolCribTab(container);
      return;
    }

    if (state.activeTab === "chipload") {
      const policyLabel =
        state.speedPolicy === "matched"
          ? "Fair Same-Speed (" + (state.baselinePhysics ? state.baselinePhysics.feed_mm_s.toFixed(1) : "—") + " mm/s)"
          : "Optimal Chipload (" + Math.round(state.aggressiveness * 100) + "% Aggr)";
      container.appendChild(
        el("div", {
          className: "section-heading",
          text: state.currentCase.case_id + " — Live Comparison: " + policyLabel,
        })
      );

      const sbsSpec = findSideBySideVariants(state.currentCase);
      const naiveT = Number(sbsSpec[0].variant.metrics.cycle_time_s || 1.0);
      const liveRows = sbsSpec.map((item, idx) => {
        const v = item.variant;
        const es = v.plan.engagement_summary || {};
        const dur = Number(v.metrics.cycle_time_s || 0.0);
        const gain = idx === 0 ? 0.0 : ((naiveT - dur) / Math.max(naiveT, 1e-6)) * 100.0;
        return {
          raw: v,
          cells: [
            item.title.replace(/^\d+\.\s*/, ""),
            dur.toFixed(2) + "s" + (idx === 0 ? "" : " (" + (gain >= 0 ? "-" : "+") + Math.abs(gain).toFixed(1) + "%)"),
            Number(es.engaged_time_s || 0).toFixed(1) + " / " + Number(es.air_cut_time_s || 0).toFixed(1) + "s",
            Number(es.mean_hex_mm || 0).toFixed(4),
            el("span", {
              className: Number(es.optimal_hex_pct || 0) >= 65.0 ? "badge-pass" : "badge-warn",
              text: Number(es.optimal_hex_pct || 0).toFixed(0) + "%",
            }),
            Number(v.plan.max_xy_shift_mm || 0).toFixed(2) + "mm",
          ],
        };
      });
      container.appendChild(
        buildTable(
          ["Approach", "Cycle (Δ)", "Eng / Air", "h_ex (mm)", "Opt%", "ΔXY"],
          liveRows,
          (r) => state.currentVariant && r.raw.id === state.currentVariant.id
        )
      );

      const agg = state.manifest.chipload_summary || [];
      if (agg.length > 0) {
        container.appendChild(
          el("div", {
            className: "section-heading",
            text: "8-Test-Part Benchmark — Matched Speed vs. Optimal Chipload (6061-T6)",
          })
        );
        const aggRows = agg.map((r) => {
          const mTime = r.mean_matched_cycle_time_s !== undefined ? Number(r.mean_matched_cycle_time_s).toFixed(1) + "s" : "—";
          return {
            raw: r,
            cells: [
              r.method,
              mTime,
              Number(r.mean_cycle_time_s).toFixed(1) + "s (-" + Number(r.mean_gain_pct).toFixed(1) + "%)",
              Number(r.mean_air_cut_time_s).toFixed(1) + "s",
              Number(r.mean_optimal_hex_pct).toFixed(1) + "%",
            ],
          };
        });
        container.appendChild(
          buildTable(
            ["Method", "Same-Spd", "Opt-Chip", "Air G1", "Opt h_ex"],
            aggRows,
            (r) => state.currentVariant && r.raw.method.indexOf("DiffCAM-Chipload") >= 0 && state.currentVariant.id === "chipload_ste"
          )
        );
      }
      return;
    }

  // --- 2-Panel Shop Setup Sheet SVG Diagram (Top XY Table + Front XZ Elevation) ---
  function buildSetupSheetDiagramSvg(sheet, setupObj) {
    const dims = sheet.stock_dims_mm || [50, 40, 15];
    const pDims = sheet.part_dims_mm || [dims[0] - 2, dims[1] - 2, dims[2] - 2];
    const lx = Math.max(12.0, Number(dims[0] || 50));
    const ly = Math.max(12.0, Number(dims[1] || 40));
    const lz = Math.max(4.0, Number(dims[2] || 15));
    const isOp2 = Number(setupObj.setup_index || 1) >= 2;
    const wcs = String(setupObj.work_offset || (isOp2 ? "G55" : "G54"));
    const jawTopZ = Number(setupObj.jaw_top_z_mm !== undefined ? setupObj.jaw_top_z_mm : -2.0);
    const sideTouchZ = Number(setupObj.side_touch_z_mm !== undefined ? setupObj.side_touch_z_mm : -3.5);
    const preZ = Number((setupObj.probe_start_xyz_mm && setupObj.probe_start_xyz_mm[2]) || 10.0);
    const xyTol = Number(setupObj.xy_jog_tolerance_mm || 5.0);
    const spSpec = (sheet.probe_package && sheet.probe_package.spindle_probe) || {};
    const prbTnum = Number(spSpec.tool_number || 99);

    // Scale stock aspect ratio cleanly inside Left Top-View panel (center x=116, y=116)
    const maxAspect = Math.max(lx / 112.0, ly / 76.0, 1e-6);
    const stockW = Math.max(58.0, Math.min(116.0, lx / maxAspect));
    const stockH = Math.max(42.0, Math.min(78.0, ly / maxAspect));
    const partW = Math.max(38.0, stockW * Math.min(0.95, Math.max(0.68, Number(pDims[0] || lx) / lx)));
    const partH = Math.max(28.0, stockH * Math.min(0.95, Math.max(0.68, Number(pDims[1] || ly) / ly)));
    const topCx = 114;
    const topCy = 115;
    const sx0 = Math.round(topCx - stockW * 0.5);
    const sy0 = Math.round(topCy - stockH * 0.5);
    const px0 = Math.round(topCx - partW * 0.5);
    const py0 = Math.round(topCy - partH * 0.5);
    const jawW = Math.round(stockW + 28);
    const jawX0 = Math.round(topCx - jawW * 0.5);

    const nodes = [];

    // Outer frame + Left/Right panel backgrounds
    nodes.push(svgEl("rect", { x: "2", y: "2", width: "536", height: "214", rx: "6", class: "setup-svg-bg" }));
    nodes.push(svgEl("rect", { x: "8", y: "24", width: "256", height: "184", rx: "4", class: "setup-svg-pane" }));
    nodes.push(svgEl("rect", { x: "274", y: "24", width: "258", height: "184", rx: "4", class: "setup-svg-pane" }));

    // Panel Titles
    nodes.push(
      svgEl("text", { x: "14", y: "16", class: "setup-svg-heading" }, [
        "TOP VIEW (XY) · " + wcs + (isOp2 ? " SOFT-JAW FLIP" : " HARD JAWS + OTS"),
      ])
    );
    nodes.push(
      svgEl("text", { x: "280", y: "16", class: "setup-svg-heading" }, [
        "FRONT ELEVATION (XZ) · T" + prbTnum + " OMP40-2 HEIGHT",
      ])
    );

    // --- LEFT PANEL: TOP VIEW (XY) ---
    //Kurt DX6 Fixed Jaw (+Y top in plan view) & Movable Jaw (-Y bottom in plan view)
    nodes.push(
      svgEl("rect", {
        x: String(jawX0),
        y: String(sy0 - 17),
        width: String(jawW),
        height: "16",
        rx: "2",
        class: isOp2 ? "setup-svg-softjaw" : "setup-svg-hardjaw",
      })
    );
    nodes.push(
      svgEl("text", { x: String(topCx), y: String(sy0 - 6), "text-anchor": "middle", class: "setup-svg-jaw-lbl" }, [
        isOp2 ? "G55 CONTOURED SOFT JAW (+Y)" : "KURT DX6 FIXED JAW (+Y REF)",
      ])
    );
    nodes.push(
      svgEl("rect", {
        x: String(jawX0),
        y: String(sy0 + Math.round(stockH) + 1),
        width: String(jawW),
        height: "15",
        rx: "2",
        class: isOp2 ? "setup-svg-softjaw" : "setup-svg-hardjaw",
      })
    );
    nodes.push(
      svgEl(
        "text",
        {
          x: String(topCx),
          y: String(sy0 + Math.round(stockH) + 12),
          "text-anchor": "middle",
          class: "setup-svg-jaw-lbl",
        },
        [isOp2 ? "MOVABLE SOFT JAW (-Y CLAMP)" : "MOVABLE JAW (-Y CLAMP ↑)"]
      )
    );

    // Workstop on -X side (Setup 1) or 180-deg X-flip axis callout (Setup 2)
    if (!isOp2) {
      nodes.push(
        svgEl("rect", {
          x: String(sx0 - 9),
          y: String(sy0 + 4),
          width: "8",
          height: String(Math.max(14, Math.round(stockH) - 8)),
          rx: "1.5",
          class: "setup-svg-stop",
        })
      );
      nodes.push(
        svgEl("text", { x: String(sx0 - 12), y: String(topCy + 3), "text-anchor": "end", class: "setup-svg-dim" }, [
          "STOP",
        ])
      );
    } else {
      nodes.push(
        svgEl("line", {
          x1: String(sx0 - 16),
          y1: String(topCy),
          x2: String(sx0 + Math.round(stockW) + 16),
          y2: String(topCy),
          class: "setup-svg-flip-axis",
        })
      );
    }

    // Stock billet & inner part outline
    nodes.push(
      svgEl("rect", {
        x: String(sx0),
        y: String(sy0),
        width: String(Math.round(stockW)),
        height: String(Math.round(stockH)),
        rx: "2",
        class: "setup-svg-stock",
      })
    );
    nodes.push(
      svgEl("rect", {
        x: String(px0),
        y: String(py0),
        width: String(Math.round(partW)),
        height: String(Math.round(partH)),
        rx: "3",
        class: "setup-svg-part",
      })
    );

    // ±5 mm manual probe jog tolerance ring & WCS origin crosshair (X=0, Y=0)
    nodes.push(
      svgEl("circle", {
        cx: String(topCx),
        cy: String(topCy),
        r: "13",
        class: "setup-svg-jog-ring",
      })
    );
    nodes.push(
      svgEl("line", {
        x1: String(topCx - 17),
        y1: String(topCy),
        x2: String(topCx + 17),
        y2: String(topCy),
        class: "setup-svg-crosshair",
      })
    );
    nodes.push(
      svgEl("line", {
        x1: String(topCx),
        y1: String(topCy - 17),
        x2: String(topCx),
        y2: String(topCy + 17),
        class: "setup-svg-crosshair",
      })
    );
    nodes.push(
      svgEl("circle", {
        cx: String(topCx),
        cy: String(topCy),
        r: "4.2",
        class: "setup-svg-ruby",
      })
    );
    nodes.push(
      svgEl("text", { x: String(topCx + 7), y: String(topCy - 6), class: "setup-svg-wcs-lbl" }, [
        wcs + " (0,0)",
      ])
    );

    // 4 OMP40-2 side-web touch points (-X, +X, -Y, +Y)
    const sidePts = [
      [sx0 - 5, topCy, sx0, topCy],
      [sx0 + Math.round(stockW) + 5, topCy, sx0 + Math.round(stockW), topCy],
      [topCx, sy0 - 5, topCx, sy0],
      [topCx, sy0 + Math.round(stockH) + 5, topCx, sy0 + Math.round(stockH)],
    ];
    for (const [cxP, cyP, txP, tyP] of sidePts) {
      nodes.push(svgEl("line", { x1: String(cxP), y1: String(cyP), x2: String(txP), y2: String(tyP), class: "setup-svg-touch-line" }));
      nodes.push(svgEl("circle", { cx: String(cxP), cy: String(cyP), r: "2.8", class: "setup-svg-ruby" }));
    }

    // Billet XY dimension label
    nodes.push(
      svgEl("text", { x: String(topCx), y: "199", "text-anchor": "middle", class: "setup-svg-note" }, [
        lx.toFixed(1) + " × " + ly.toFixed(1) + " mm · Jog XY ±" + xyTol.toFixed(0) + " mm",
      ])
    );

    // Renishaw OTS Optical Table Tool Setter on right (+X) side of machine table
    nodes.push(
      svgEl("rect", { x: "212", y: "66", width: "44", height: "86", rx: "4", class: "setup-svg-ots-base" })
    );
    nodes.push(svgEl("circle", { cx: "234", cy: "95", r: "10", class: "setup-svg-ots-disc" }));
    nodes.push(svgEl("line", { x1: "218", y1: "122", x2: "250", y2: "122", class: "setup-svg-optical" }));
    nodes.push(
      svgEl("text", { x: "234", y: "137", "text-anchor": "middle", class: "setup-svg-ots-lbl" }, ["OTS"])
    );
    nodes.push(
      svgEl("text", { x: "234", y: "147", "text-anchor": "middle", class: "setup-svg-dim" }, ["+X TBL"])
    );

    // --- RIGHT PANEL: FRONT ELEVATION (XZ) ---
    const elCx = 403;
    const z0Y = 114; // Top datum Z = 0.0
    const stockPxH = Math.max(28, Math.min(48, Math.round(lz * 1.9)));
    const stockBotY = z0Y + stockPxH;
    const bedY = stockBotY + 18;
    const scaleZ = stockPxH / Math.max(lz, 1.0);
    const jawTopY = Math.max(z0Y + 5, Math.min(stockBotY - 3, Math.round(z0Y + Math.abs(jawTopZ) * scaleZ)));
    const sideTouchY = Math.max(z0Y + 3, Math.min(jawTopY - 4, Math.round(z0Y + Math.abs(sideTouchZ) * scaleZ)));
    const probeBallCy = z0Y - 24;

    // Vise bed
    nodes.push(svgEl("rect", { x: "288", y: String(bedY), width: "230", height: "14", rx: "2", class: "setup-svg-visebed" }));
    nodes.push(
      svgEl("text", { x: String(elCx), y: String(bedY + 10), "text-anchor": "middle", class: "setup-svg-jaw-lbl" }, [
        "KURT DX6 PRECISION VISE BED",
      ])
    );

    // Parallels (Op1) or Soft-Jaw Seat Step (Op2) + Left/Right Jaws in XZ view
    const stLeft = elCx - 48;
    const stRight = elCx + 48;
    if (!isOp2) {
      nodes.push(svgEl("rect", { x: String(stLeft + 2), y: String(stockBotY), width: "10", height: "18", class: "setup-svg-parallel" }));
      nodes.push(svgEl("rect", { x: String(stRight - 12), y: String(stockBotY), width: "10", height: "18", class: "setup-svg-parallel" }));
    } else {
      nodes.push(svgEl("rect", { x: String(stLeft - 24), y: String(stockBotY), width: "144", height: "18", class: "setup-svg-softjaw" }));
    }
    nodes.push(
      svgEl("rect", {
        x: String(stLeft - 26),
        y: String(jawTopY),
        width: "26",
        height: String(bedY - jawTopY),
        rx: "2",
        class: isOp2 ? "setup-svg-softjaw" : "setup-svg-hardjaw",
      })
    );
    nodes.push(
      svgEl("rect", {
        x: String(stRight),
        y: String(jawTopY),
        width: "26",
        height: String(bedY - jawTopY),
        rx: "2",
        class: isOp2 ? "setup-svg-softjaw" : "setup-svg-hardjaw",
      })
    );

    // Stock billet in XZ
    nodes.push(
      svgEl("rect", {
        x: String(stLeft),
        y: String(z0Y),
        width: "96",
        height: String(stockPxH),
        rx: "2",
        class: "setup-svg-stock",
      })
    );
    if (isOp2) {
      // Highlight carrier hat at top of flipped Op2 part
      nodes.push(
        svgEl("rect", {
          x: String(stLeft),
          y: String(z0Y),
          width: "96",
          height: "8",
          class: "setup-svg-hat",
        })
      );
    }

    // Z = 0.0 Top Datum dashed line
    nodes.push(svgEl("line", { x1: "284", y1: String(z0Y), x2: "524", y2: String(z0Y), class: "setup-svg-datum-line" }));
    nodes.push(
      svgEl("text", { x: "286", y: String(z0Y - 4), class: "setup-svg-wcs-lbl" }, [wcs + " Z = 0.0"])
    );

    // Safe Side Web Touch Z marker & ruby side ball
    nodes.push(
      svgEl("line", {
        x1: String(stRight),
        y1: String(sideTouchY),
        x2: "522",
        y2: String(sideTouchY),
        class: "setup-svg-sidez-line",
      })
    );
    nodes.push(
      svgEl("circle", { cx: String(stRight + 4), cy: String(sideTouchY), r: "3.2", class: "setup-svg-ruby" })
    );
    nodes.push(
      svgEl("text", { x: "522", y: String(sideTouchY - 3), "text-anchor": "end", class: "setup-svg-dim" }, [
        "Web Z " + sideTouchZ.toFixed(2) + " > Jaw " + jawTopZ.toFixed(2),
      ])
    );

    // T99 Renishaw OMP40-2 Spindle Probe Assembly hovering at Z = +10 mm above top center
    nodes.push(svgEl("rect", { x: String(elCx - 16), y: "30", width: "32", height: "14", rx: "2", class: "setup-svg-probe-body" }));
    nodes.push(svgEl("rect", { x: String(elCx - 16), y: "44", width: "32", height: "4", class: "setup-svg-optical" }));
    nodes.push(svgEl("rect", { x: String(elCx - 2), y: "48", width: "4", height: String(probeBallCy - 48), class: "setup-svg-stem" }));
    nodes.push(svgEl("circle", { cx: String(elCx), cy: String(probeBallCy), r: "4.6", class: "setup-svg-ruby" }));
    nodes.push(
      svgEl("line", {
        x1: String(elCx),
        y1: String(probeBallCy + 6),
        x2: String(elCx),
        y2: String(z0Y - 2),
        class: "setup-svg-touch-line",
      })
    );
    nodes.push(
      svgEl("text", { x: String(elCx + 21), y: "41", class: "setup-svg-wcs-lbl" }, ["T" + prbTnum + " OMP40-2"])
    );
    nodes.push(
      svgEl("text", { x: String(elCx + 10), y: String(probeBallCy + 3), class: "setup-svg-note" }, [
        "Start Z = +" + preZ.toFixed(0) + " mm (+" + (preZ / INCH_MM).toFixed(2) + '")',
      ])
    );
    nodes.push(
      svgEl("text", { x: String(elCx), y: "199", "text-anchor": "middle", class: "setup-svg-note" }, [
        isOp2
          ? "After M00: Flip 180° around X → Re-Probe G55 Top & XY"
          : "Pre-Run: Jog Ruby Tip +10 mm Over Billet Center (±5 mm)",
      ])
    );

    return svgEl("svg", { viewBox: "0 0 540 218", class: "setup-svg", role: "img", "aria-label": "Part and OMP40-2 probe setup diagram" }, nodes);
  }

  function renderSetupSheetAndToolsSection(container) {
    const sheet = resolveCaseSetupSheet(state.currentCase, state.currentVariant);
    if (!sheet) return;

    const setups = sheet.setups || [];
    if (state.setupSheetActiveIdx > setups.length) {
      state.setupSheetActiveIdx = 1;
    }
    const activeSetup =
      setups.find((s) => Number(s.setup_index) === Number(state.setupSheetActiveIdx)) || setups[0];

    const pkg = sheet.probe_package || {};
    const progCode = "O" + String(Number(sheet.program_number || 2001)).padStart(5, "0");
    const sDims = sheet.stock_dims_mm || [0, 0, 0];

    const copySheetBtn = el("button", {
      type: "button",
      className: "btn btn-tiny",
      text: "Copy Setup Sheet",
    });
    copySheetBtn.addEventListener("click", async () => {
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(formatSetupSheetMarkdown(sheet));
          copySheetBtn.textContent = "Copied ✓";
          setTimeout(() => {
            copySheetBtn.textContent = "Copy Setup Sheet";
          }, 1400);
        }
      } catch (_err) {
        copySheetBtn.textContent = "Copy N/A";
      }
    });

    const probeOp = (state.operations || []).find(
      (o) =>
        Number(o.setupIndex || 1) === Number(activeSetup.setup_index || 1) &&
        (String(o.kind || "").indexOf("probe") >= 0 || Number(o.toolNumber) >= 90)
    );
    const headerActions = [copySheetBtn];
    if (probeOp) {
      const scrubProbeBtn = el("button", {
        type: "button",
        className: "btn btn-tiny btn-accent",
        text: "Scrub " + activeSetup.work_offset + " Probe",
      });
      scrubProbeBtn.addEventListener("click", () => scrubToOperation(probeOp, false));
      headerActions.unshift(scrubProbeBtn);
    }

    // 1. Setup Sheet Header Banner
    container.appendChild(
      el("div", { className: "setup-sheet-card" }, [
        el("div", { className: "setup-card-header" }, [
          el("div", null, [
            el("div", {
              className: "setup-card-title",
              text: progCode + " · " + sheet.part_title + " — Shop Setup Sheet",
            }),
            el("div", {
              className: "setup-card-sub",
              text:
                String(pkg.package_label || "Haas WIPS-R (Renishaw OMP40-2 + OTS)") +
                " · Billet " +
                Number(sDims[0]).toFixed(1) +
                " × " +
                Number(sDims[1]).toFixed(1) +
                " × " +
                Number(sDims[2]).toFixed(1) +
                " mm",
            }),
          ]),
          el("div", { className: "op-row-actions" }, headerActions),
        ]),

        // Setup 1 (G54) / Setup 2 (G55 Post-Flip) Selector Pills
        setups.length > 1
          ? el(
              "div",
              { className: "setup-stage-tabs" },
              setups.map((st) => {
                const isSel = Number(st.setup_index) === Number(activeSetup.setup_index);
                const b = el("button", {
                  type: "button",
                  className: isSel ? "btn btn-tiny btn-active" : "btn btn-tiny",
                  text:
                    "Setup " +
                    st.setup_index +
                    " (" +
                    st.work_offset +
                    "): " +
                    (Number(st.setup_index) >= 2 ? "Post-Flip Re-Probe" : "Initial Step-1 Probe"),
                });
                b.addEventListener("click", () => {
                  state.setupSheetActiveIdx = Number(st.setup_index);
                  const targetProbe = (state.operations || []).find(
                    (o) => Number(o.setupIndex || 1) === Number(st.setup_index)
                  );
                  if (targetProbe) {
                    scrubToOperation(targetProbe, false);
                  }
                  renderInspectorTab();
                });
                return b;
              })
            )
          : el("div", { className: "setup-stage-badge-row" }, [
              el("span", { className: "ots-badge ots-badge-probe", text: activeSetup.stage_badge || "STEP 1 · G54 PROBE" }),
              el("span", { className: "setup-stage-ref", text: activeSetup.vise_configuration || "" }),
            ]),

        // Key Part-to-Probe Positioning Coordinates Grid
        el("div", { className: "setup-kv-grid" }, [
          el("div", { className: "setup-kv-box" }, [
            el("span", { className: "setup-kv-label", text: activeSetup.work_offset + " WCS Datum Origin" }),
            el("span", { className: "setup-kv-val", text: "Top-Center (X0, Y0, Z0)" }),
          ]),
          el("div", { className: "setup-kv-box" }, [
            el("span", { className: "setup-kv-label", text: "Pre-Run OMP40-2 Tip Pos" }),
            el("span", {
              className: "setup-kv-val setup-kv-cyan",
              text:
                "X0 Y0 (±" +
                Number(activeSetup.xy_jog_tolerance_mm || 5) +
                "mm), Z +" +
                Number((activeSetup.probe_start_xyz_mm || [0, 0, 10])[2]).toFixed(0) +
                "mm",
            }),
          ]),
          el("div", { className: "setup-kv-box" }, [
            el("span", { className: "setup-kv-label", text: "Side Web Touch Z (Safe)" }),
            el("span", {
              className: "setup-kv-val setup-kv-good",
              text:
                "Z " +
                Number(activeSetup.side_touch_z_mm || -3.5).toFixed(2) +
                " mm (> Jaw " +
                Number(activeSetup.jaw_top_z_mm || -5.0).toFixed(2) +
                ")",
            }),
          ]),
        ]),

        // 2-Panel Positioning Diagram (Top XY + Front XZ)
        el("div", { className: "setup-diagram-wrap" }, [buildSetupSheetDiagramSvg(sheet, activeSetup)]),

        // Operator Pre-Start Positioning Checklist
        el(
          "ol",
          { className: "setup-checklist" },
          (activeSetup.checklist || []).map((line) => el("li", { text: String(line) }))
        ),
      ])
    );

    // 2. Combined Per-Tool List & Renishaw OTS Calibration Schedule
    const toolsSched = sheet.tools_schedule || [];
    if (toolsSched.length > 0) {
      container.appendChild(
        el("div", {
          className: "section-heading",
          text: "Program Tool List & Renishaw OTS Offsets (" + toolsSched.length + " Tools)",
        })
      );
      const toolRows = toolsSched.map((t) => {
        const isPrb = t.ots_mode === "spindle_probe_master" || t.entity_type === "probe";
        const badgeCls = isPrb
          ? "ots-badge ots-badge-probe"
          : t.ots_mode === "ots_length_and_diameter"
            ? "ots-badge ots-badge-rot"
            : "ots-badge ots-badge-len";
        const badgeShort = isPrb
          ? "OMP40-2 WCS"
          : t.ots_mode === "ots_length_and_diameter"
            ? "OTS Len+Dia (B3)"
            : "OTS Len (B1)";

        const firstOpForTool = (state.operations || []).find(
          (o) => Number(o.toolNumber) === Number(t.tool_number)
        );
        const tBtn = el("button", {
          type: "button",
          className: "btn btn-tiny",
          text: "T" + t.tool_number,
          title: firstOpForTool ? "Scrub to first operation with T" + t.tool_number : t.offset_registers,
        });
        if (firstOpForTool) {
          tBtn.addEventListener("click", () => scrubToOperation(firstOpForTool, false));
        }

        return {
          raw: t,
          cells: [
            el("div", { className: "tool-sched-id" }, [
              tBtn,
              el("span", { className: "tool-sched-reg", text: t.offset_registers }),
            ]),
            el("div", { className: "tool-sched-desc" }, [
              el("div", { className: "tool-sched-name", text: t.label }),
              el("div", {
                className: "tool-sched-sub",
                text: t.holder_label + " · Used: " + (t.used_in_ops || []).join(", "),
              }),
            ]),
            "Ø" +
              Number(t.diameter_mm).toFixed(2) +
              "mm (" +
              Number(t.diameter_in).toFixed(3) +
              '") · Out ' +
              Number(t.stickout_mm).toFixed(1) +
              "mm",
            el("div", { className: "tool-sched-ots" }, [
              el("span", { className: badgeCls, text: badgeShort }),
              el("div", { className: "tool-sched-macro", text: t.ots_macro_preview }),
            ]),
          ],
        };
      });

      container.appendChild(
        buildTable(
          ["Tool / Reg", "Tool & Holder (Used In)", "Dia / Stickout", "WIPS-R Calibration"],
          toolRows,
          (r) => {
            const activeOp = state.opByKey.get(state.activeOpKey);
            return activeOp && Number(activeOp.toolNumber) === Number(r.raw.tool_number);
          }
        )
      );
    }
  }

    if (state.activeTab === "case") {
      renderSetupSheetAndToolsSection(container);

      const wh = state.workholding || resolveCaseWorkholding(state.currentCase);
      if (wh && wh.setups) {
        const s1 = wh.setups["1"];
        const s2 = wh.setups["2"];
        const hasOp2 = Boolean(
          wh.has_op2_soft_jaws || (state.operations || []).some((o) => o.setupIndex >= 2)
        );
        container.appendChild(
          el("div", {
            className: "section-heading",
            text: "TAMP Workholding — " + String(wh.vise_label || 'Kurt DX6 CrossOver 6" Machine Vise'),
          })
        );
        const whRows = [
          {
            cells: [
              "Vise & Jaw Mode",
              String(wh.clamping_axis || "Y") +
                "-Axis Clamp · " +
                (wh.jaw_mounting === "reversed_exterior" ? "Reversed Exterior" : "Standard Interior"),
              Number(wh.jaw_width_mm || 152.4).toFixed(1) + " × " + Number(wh.jaw_height_mm || 44.45).toFixed(2) + " mm",
            ],
          },
          {
            cells: [
              "Op1 (G54)",
              "Hardened Steel Jaws + " + Number((s1 && s1.parallel_height_mm) || 42.45).toFixed(2) + " mm Parallels",
              "Open " +
                Number((s1 && s1.jaw_opening_mm) || 0).toFixed(1) +
                " mm · Bite " +
                Number((s1 && s1.grip_z_mm) || 2.0).toFixed(1) +
                " mm",
            ],
          },
        ];
        if (hasOp2 && s2) {
          whRows.push({
            cells: [
              "Op2 (G55 Flip)",
              "Machined 6061-T6 Contoured Soft Jaws",
              "Open " +
                Number(s2.jaw_opening_mm || 0).toFixed(1) +
                " mm · Step Z " +
                Number(s2.seat_z_mm || 0).toFixed(1) +
                " mm",
            ],
          });
        }
        container.appendChild(buildTable(["Stage", "Workholding Configuration", "Clamping Span"], whRows));
      }

      // 2. Multi-Tool CAD B-Rep Summary when present
      const cadMeta = state.currentCase.cad_metadata || null;
      if (cadMeta) {
        container.appendChild(
          el("div", {
            className: "section-heading",
            text: "CAD B-Rep & 6-Axis Recognition Summary",
          })
        );
        const metaRows = [
          {
            cells: [
              "CAD Units / Setup",
              String(cadMeta.source_units || "mm") + " → " + String(cadMeta.setup_summary || "G54 Op1"),
              String(cadMeta.bbox_mm_str || ""),
            ],
          },
          {
            cells: [
              "Recognized Features",
              String(cadMeta.features_summary || "Prismatic + Holes"),
              String(cadMeta.unit_library_label || ""),
            ],
          },
        ];
        container.appendChild(buildTable(["Attribute", "Planned Configuration", "Size / Crib"], metaRows));

        if (Array.isArray(cadMeta.warnings) && cadMeta.warnings.length > 0) {
          for (const w of cadMeta.warnings) {
            container.appendChild(
              el("div", { className: "status-banner", text: "Tool Crib Note: " + String(w) })
            );
          }
        }
      }

      // 3. Multi-Operation Interactive Table (Toggle multiple ops, Solo, or Scrub Start/End)
      const derivedOps = state.operations || [];
      if (derivedOps.length > 0) {
        const isAll =
          state.opFilterMode !== "custom" || state.selectedOpKeys.size >= derivedOps.length;
        container.appendChild(
          el("div", {
            className: "section-heading",
            text:
              "Operations (" +
              (isAll ? "All " + derivedOps.length : state.selectedOpKeys.size + " / " + derivedOps.length) +
              " Visible — Check to Overlay Multiple)",
          })
        );

        const opRows = derivedOps.map((op) => {
          const vis = isAll || state.selectedOpKeys.has(op.key);
          const chk = el("input", {
            type: "checkbox",
            className: "op-row-chk",
            "aria-label": "Show Op " + op.opIndex,
          });
          chk.checked = vis;
          chk.addEventListener("change", () => {
            const nextSet = new Set(isAll ? derivedOps.map((o) => o.key) : state.selectedOpKeys);
            if (chk.checked) nextSet.add(op.key);
            else nextSet.delete(op.key);
            applyOpSelectionSet(nextSet, false);
          });

          const btnStart = el(
            "button",
            { type: "button", className: "btn btn-tiny", text: "Scrub", title: "Jump playhead to operation start" }
          );
          btnStart.addEventListener("click", () => scrubToOperation(op, false));

          const btnSolo = el(
            "button",
            { type: "button", className: "btn btn-tiny", text: "Solo", title: "Isolate only this operation's toolpath" }
          );
          btnSolo.addEventListener("click", () => applyOpSelectionSet(new Set([op.key]), true));

          return {
            raw: op,
            cells: [
              el("span", { className: "op-row-actions" }, [
                chk,
                el("span", { className: "op-swatch op-swatch-" + op.colorIdx }),
                el("strong", { className: "accent-mono", text: "Op" + op.opIndex }),
              ]),
              op.workOffset + " · T" + op.toolNumber + " " + op.shortKind,
              op.durationS.toFixed(1) + "s",
              el("span", { className: "op-row-actions" }, [btnStart, btnSolo]),
            ],
          };
        });

        container.appendChild(
          buildTable(
            ["Show", "Setup / Tool / Op", "Dur.", "Inspect"],
            opRows,
            (r) => r.raw.key === state.activeOpKey
          )
        );
      }

      if ((state.currentCase.benchmark_table || []).length > 0) {
        container.appendChild(
          el("div", {
            className: "section-heading",
            text: state.currentCase.case_id + " (" + state.currentCase.family + ") — Benchmark Summary",
          })
        );
        const activeMethod = state.currentVariant ? state.currentVariant.method : "";
        const rows = (state.currentCase.benchmark_table || []).map((r) => ({
          raw: r,
          cells: [
            r.method,
            r.cycle_time_mean_s.toFixed(2) + " ± " + r.cycle_time_std_s.toFixed(2),
            (r.gain_mean_pct >= 0 ? "-" : "+") + Math.abs(r.gain_mean_pct).toFixed(1) + "%",
            r.peak_mrr_mean_mm3_s.toFixed(1),
            el("span", {
              className: r.feasible_rate_pct >= 99.9 ? "badge-pass" : "badge-warn",
              text: r.feasible_rate_pct.toFixed(0) + "%",
            }),
          ],
        }));
        container.appendChild(
          buildTable(
            ["Method", "Cycle (s)", "Time Δ", "Peak MRR", "Feas."],
            rows,
            (r) => r.raw.method === activeMethod
          )
        );
      }

      container.appendChild(
        el("div", { className: "section-heading", text: "Posted G-Code & Dexel Verification" })
      );
      const vRows = state.currentCase.variants.map((v) => ({
        raw: v,
        cells: [
          v.label,
          v.plan.num_segments,
          v.metrics.orig_gouge_mm3.toFixed(4) + " mm³",
          v.metrics.residual_mm3.toFixed(4) + " mm³",
          el("span", {
            className: v.metrics.posted_geometry_pass && v.metrics.posted_safety_pass ? "badge-pass" : "badge-warn",
            text: v.metrics.posted_geometry_pass && v.metrics.posted_safety_pass ? "PASS" : "FAIL",
          }),
        ],
      }));
      container.appendChild(
        buildTable(
          ["Simulated Variant", "Segs", "Gouge", "Residual", "Oracle"],
          vRows,
          (r) => state.currentVariant && r.raw.id === state.currentVariant.id
        )
      );
      return;
    }

    const summary = state.manifest.paper_summary || {};
    container.appendChild(
      el("div", {
        className: "section-heading",
        text: "Table I — 16-Task Main Benchmark (64 Evaluations x 3 Seeds)",
      })
    );
    const exp2Main = summary.exp2_main || {};
    const mRows = Object.entries(exp2Main).map(([method, d]) => ({
      method,
      cells: [
        method,
        Number(d.mean_cycle_time_s).toFixed(2) + "s",
        "-" + Number(d.overall_gain_mean).toFixed(1) + " ± " + Number(d.overall_gain_std).toFixed(1) + "%",
        Number(d.mean_peak_mrr_mm3_s).toFixed(1),
        el("span", {
          className: Number(d.feasible_rate_pct) >= 99.0 ? "badge-pass" : "badge-warn",
          text: Number(d.feasible_rate_pct).toFixed(1) + "%",
        }),
      ],
    }));
    container.appendChild(
      buildTable(
        ["Method", "Mean Cycle", "Cycle Gain", "Peak MRR", "Feas."],
        mRows,
        (r) => state.currentVariant && r.method === state.currentVariant.method
      )
    );

    container.appendChild(
      el("div", { className: "section-heading", text: "Table II — Residual Policy Transfer (Held-Out Test)" })
    );
    const exp3 = summary.exp3_transfer || {};
    const tRows = Object.entries(exp3).map(([reg, d]) => ({
      cells: [
        reg,
        "-" + Number(d.gain_mean).toFixed(1) + " ± " + Number(d.gain_std).toFixed(1) + "%",
        Number(d.peak_mrr_mean).toFixed(1),
        el("span", {
          className: Number(d.feasible_pct) >= 99.0 ? "badge-pass" : "badge-warn",
          text: Number(d.feasible_pct).toFixed(1) + "%",
        }),
      ],
    }));
    container.appendChild(buildTable(["Transfer Regime", "Cycle Gain", "Peak MRR", "Feas."], tRows));

    const mc = (summary.exp4_verification || {}).multi_controller_summary || {};
    if (mc.by_dialect) {
      container.appendChild(
        el("div", { className: "section-heading", text: "6-Controller Clean-Room Post Audit (Exp4)" })
      );
      const dRows = Object.entries(mc.by_dialect).map(([dial, d]) => ({
        cells: [
          dial.toUpperCase(),
          d.pass_count + " / " + d.count,
          Number(d.mean_cycle_time_s).toFixed(2) + " s",
          el("span", {
            className: d.pass_count === d.count ? "badge-pass" : "badge-warn",
            text: d.pass_count === d.count ? "100% PASS" : "CHECK",
          }),
        ],
      }));
      container.appendChild(buildTable(["Controller", "Geom+Safe", "Mean Cycle", "Status"], dRows));
    }
  }

  // --- Main WebGL Render Loop (Single & 3-Pane Scissored Side-by-Side) ---
  function drawMeshBuffer(posBuf, nrmBuf, colBuf, count, mvp, modelMat, eye, baseColor, useVertCol, alpha, unlitMix) {
    if (count <= 0) return;
    const gl = renderer.gl;
    const m = renderer.mesh;
    gl.useProgram(m.prog);
    gl.uniformMatrix4fv(m.uMVP, false, mvp);
    gl.uniformMatrix4fv(m.uModel, false, modelMat);
    gl.uniform3fv(m.uBaseColor, baseColor);
    gl.uniform1f(m.uUseVertColor, useVertCol ? 1.0 : 0.0);
    gl.uniform3fv(m.uEye, eye);
    gl.uniform1f(m.uAlpha, alpha);
    gl.uniform1f(m.uUnlitMix, unlitMix || 0.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    if (useVertCol && colBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(2);
      gl.vertexAttrib3f(2, baseColor[0], baseColor[1], baseColor[2]);
    }

    gl.drawArrays(gl.TRIANGLES, 0, count);
  }

  function drawLineBuffer(posBuf, colBuf, firstVertex, vertexCount, mvp) {
    if (vertexCount <= 0) return;
    const gl = renderer.gl;
    const l = renderer.line;
    gl.useProgram(l.prog);
    gl.uniformMatrix4fv(l.uMVP, false, mvp);

    gl.disableVertexAttribArray(2);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.LINES, firstVertex, vertexCount);
  }

  function countVisibleSubset(segIndices, maxSegIndex) {
    if (maxSegIndex < 0) return 0;
    let lo = 0;
    let hi = segIndices.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (segIndices[mid] <= maxSegIndex) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function renderPaneViewport(slot, x0, y0, pw, ph, eye, view, flags) {
    const gl = renderer.gl;
    gl.viewport(x0, y0, pw, ph);
    gl.scissor(x0, y0, pw, ph);
    gl.clearColor(0.055, 0.075, 0.105, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const proj = mat4Perspective((40.0 * Math.PI) / 180.0, pw / Math.max(ph, 1), 0.5, 4200.0);
    const vp = mat4Multiply(proj, view);
    const ident = mat4Identity();
    const workpieceModel =
      slot.activeSetupIndex === 2 && state.currentCase
        ? mat4FlipX180(getCaseZFlipSum(state.currentCase))
        : ident;

    const effectiveStockAlpha =
      state.stockMode === "live_glass" ? Math.min(state.stockAlpha, 0.52) : state.stockAlpha;
    const stockOpaque = effectiveStockAlpha >= 0.995;
    const useStaticStl = state.stockMode === "static_stl";
    const sPos = useStaticStl ? slot.staticStockPos : slot.liveStockPos;
    const sNrm = useStaticStl ? slot.staticStockNrm : slot.liveStockNrm;
    const sCol = useStaticStl ? slot.staticStockCol : slot.liveStockCol;
    const sCnt = useStaticStl ? slot.staticStockCount : slot.liveStockCount;

    gl.disable(gl.BLEND);
    gl.depthMask(true);

    // 0. Stationary TAMP-Selected Kurt DX6 CrossOver 6" Machine Vise (G54 Hard Jaws + Parallels vs G55 Soft Jaws)
    if (flags.showVise) {
      const useSetup2Vise = slot.activeSetupIndex >= 2 && state.vise2Count > 0;
      const vPos = useSetup2Vise ? renderer.shared.vise2Pos : renderer.shared.vise1Pos;
      const vNrm = useSetup2Vise ? renderer.shared.vise2Nrm : renderer.shared.vise1Nrm;
      const vCol = useSetup2Vise ? renderer.shared.vise2Col : renderer.shared.vise1Col;
      const vCnt = useSetup2Vise ? state.vise2Count : state.vise1Count;
      if (vCnt > 0) {
        drawMeshBuffer(vPos, vNrm, vCol, vCnt, vp, ident, eye, [0.12, 0.36, 0.66], true, 1.0, 0.0);
      }
    }

    // 1. Solid Target CAD underneath in Glass mode
    if (state.stockMode === "live_glass" && state.targetCount > 0) {
      drawMeshBuffer(
        renderer.shared.targetPos,
        renderer.shared.targetNrm,
        null,
        state.targetCount,
        vp,
        workpieceModel,
        eye,
        [0.72, 0.80, 0.88],
        false,
        1.0,
        0.0
      );
    }

    // 2. Opaque Live Cutting Stock (Flips 180° around X during Op2 G55!)
    if (flags.showStock && stockOpaque && sCnt > 0) {
      drawMeshBuffer(sPos, sNrm, sCol, sCnt, vp, workpieceModel, eye, [0.72, 0.78, 0.85], true, 1.0, 0.0);
    }

    // 3. Thick 3D Extruded Toolpath Ribbons (unlitMix = 0.82 preserves vivid tight-range colors!)
    const maxIdx = flags.clipToTime ? slot.activeSegIndex : 1000000000;
    if (flags.showRapids && slot.rapidTotalVerts > 0) {
      const nRapVerts = flags.clipToTime
        ? countVisibleSubset(slot.rapidSegIndices, maxIdx) * slot.rapidVertsPerSeg
        : slot.rapidTotalVerts;
      drawMeshBuffer(
        slot.rapidPos,
        slot.rapidNrm,
        slot.rapidCol,
        nRapVerts,
        vp,
        workpieceModel,
        eye,
        [0.96, 0.65, 0.14],
        true,
        1.0,
        0.80
      );
    }
    if (flags.showCuts && slot.cutTotalVerts > 0) {
      const nCutVerts = flags.clipToTime
        ? countVisibleSubset(slot.cutSegIndices, maxIdx) * slot.cutVertsPerSeg
        : slot.cutTotalVerts;
      drawMeshBuffer(
        slot.cutPos,
        slot.cutNrm,
        slot.cutCol,
        nCutVerts,
        vp,
        workpieceModel,
        eye,
        [0.2, 0.8, 0.5],
        true,
        1.0,
        0.84
      );
    }
    if (slot.activeVerts > 0 && (flags.showCuts || flags.showRapids)) {
      drawMeshBuffer(
        slot.activePos,
        slot.activeNrm,
        slot.activeCol,
        slot.activeVerts,
        vp,
        workpieceModel,
        eye,
        [1.0, 1.0, 1.0],
        true,
        1.0,
        0.92
      );
    }

    // 4. Dynamic Tool-Library-Linked 3D Cutter + Shank + Holder (Upright +Z Spindle)
    const tGpu = slot.activeToolGpu;
    const tPosBuf = tGpu ? tGpu.pos : renderer.shared.toolPos;
    const tNrmBuf = tGpu ? tGpu.nrm : renderer.shared.toolNrm;
    const tColBuf = tGpu ? tGpu.col : renderer.shared.toolCol;
    const tVertCnt = tGpu ? tGpu.count : state.toolCount;
    if (flags.showTool && tVertCnt > 0) {
      const toolModel = mat4Translation(slot.toolPos[0], slot.toolPos[1], slot.toolPos[2]);
      drawMeshBuffer(
        tPosBuf,
        tNrmBuf,
        tColBuf,
        tVertCnt,
        vp,
        toolModel,
        eye,
        [1.0, 1.0, 1.0],
        true,
        1.0,
        0.0
      );
    }

    // 5. Reference Grid + G54 Axes + Billet Box Wireframe
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    if (state.guideGridCount > 0) {
      drawLineBuffer(renderer.shared.guidePos, renderer.shared.guideCol, 0, state.guideGridCount, vp);
    }
    if (flags.showBillet && state.guideBilletCount > 0) {
      drawLineBuffer(
        renderer.shared.guidePos,
        renderer.shared.guideCol,
        state.guideBilletOffset,
        state.guideBilletCount,
        vp
      );
    }

    // 6. Translucent Stock / Target CAD Overlay
    gl.depthMask(false);
    if (flags.showStock && !stockOpaque && sCnt > 0) {
      drawMeshBuffer(
        sPos,
        sNrm,
        sCol,
        sCnt,
        vp,
        workpieceModel,
        eye,
        [0.22, 0.56, 0.26],
        true,
        effectiveStockAlpha,
        0.0
      );
    }
    if (flags.showTarget && state.stockMode !== "live_glass" && state.targetCount > 0) {
      drawMeshBuffer(
        renderer.shared.targetPos,
        renderer.shared.targetNrm,
        null,
        state.targetCount,
        vp,
        workpieceModel,
        eye,
        [0.18, 0.85, 0.64],
        false,
        0.34,
        0.0
      );
    }
    gl.depthMask(true);
  }

  function renderFrame(nowMs) {
    requestAnimationFrame(renderFrame);
    if (!renderer) return;

    if (state.playing && state.totalDurationS > 1e-6) {
      const dt = Math.min((nowMs - state.lastFrameMs) * 0.001, 0.1);
      let nextT = state.playTimeS + dt * state.speed;
      // When custom multi-operation filter is active, skip unselected operation intervals smoothly!
      if (state.opFilterMode === "custom" && state.selectedOpKeys.size > 0 && state.operations.length > 0) {
        const selOps = state.operations.filter((o) => state.selectedOpKeys.has(o.key));
        if (selOps.length > 0) {
          const lastSelEnd = selOps[selOps.length - 1].t1S;
          if (nextT >= lastSelEnd) {
            nextT = lastSelEnd;
            state.playing = false;
            document.getElementById("btn-play").textContent = "Play";
          } else {
            const insideAny = selOps.some((o) => nextT >= o.t0S - 1e-4 && nextT <= o.t1S + 1e-4);
            if (!insideAny) {
              const upcoming = selOps.find((o) => o.t0S > nextT - 1e-4);
              if (upcoming) {
                nextT = upcoming.t0S;
              } else {
                nextT = lastSelEnd;
                state.playing = false;
                document.getElementById("btn-play").textContent = "Play";
              }
            }
          }
        }
      }
      state.playTimeS = nextT;
      if (state.playTimeS >= state.totalDurationS) {
        state.playTimeS = state.totalDurationS;
        state.playing = false;
        document.getElementById("btn-play").textContent = "Play";
      }
      updateScrubberAndDro();
    }
    state.lastFrameMs = nowMs;

    const canvas = renderer.gl.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const gl = renderer.gl;
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.SCISSOR_TEST);

    const distScale = state.layoutMode === "sbs" ? 1.18 : 1.0;
    const effDist = state.cam.distance * distScale;
    const cp = Math.cos(state.cam.pitch);
    const sp = Math.sin(state.cam.pitch);
    const cy = Math.cos(state.cam.yaw);
    const sy = Math.sin(state.cam.yaw);
    const eye = [
      state.cam.target[0] + effDist * cp * cy,
      state.cam.target[1] + effDist * cp * sy,
      state.cam.target[2] + effDist * sp,
    ];
    const view = mat4LookAt(eye, state.cam.target, [0.0, 0.0, 1.0]);

    const chkViseEl = document.getElementById("chk-vise");
    const flags = {
      showVise: chkViseEl ? chkViseEl.checked : true,
      showStock: document.getElementById("chk-stock").checked,
      showTarget: document.getElementById("chk-target").checked || state.stockMode === "live_glass",
      showBillet: document.getElementById("chk-billet").checked,
      showCuts: document.getElementById("chk-cuts").checked,
      showRapids: document.getElementById("chk-rapids").checked,
      showTool: document.getElementById("chk-tool").checked,
      clipToTime: document.getElementById("chk-clip").checked,
    };

    if (state.layoutMode === "sbs") {
      const gap = 2;
      const paneW = Math.max(1, Math.floor((w - 2 * gap) / 3));
      gl.viewport(0, 0, w, h);
      gl.scissor(0, 0, w, h);
      gl.clearColor(0.16, 0.22, 0.30, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      for (let i = 0; i < 3; i++) {
        const x0 = i * (paneW + gap);
        const curW = i === 2 ? Math.max(1, w - x0) : paneW;
        renderPaneViewport(renderer.panes[i], x0, 0, curW, h, eye, view, flags);
      }
    } else {
      renderPaneViewport(renderer.panes[0], 0, 0, w, h, eye, view, flags);
    }

    gl.disable(gl.SCISSOR_TEST);
  }

  // --- Wire Mouse / Transport / Physics / Layout Controls ---
  function stepMove(delta) {
    const refSlot = state.layoutMode === "sbs" ? renderer.panes[2] : renderer.panes[0];
    if (!refSlot || !refSlot.variant) return;
    const segs = refSlot.variant.plan.segments || [];
    if (segs.length === 0) return;
    state.playing = false;
    document.getElementById("btn-play").textContent = "Play";

    const nextIdx = refSlot.activeSegIndex + delta;
    if (nextIdx < 0) {
      state.playTimeS = 0.0;
    } else if (nextIdx >= segs.length) {
      state.playTimeS = state.totalDurationS;
    } else {
      state.playTimeS = Number(segs[nextIdx].t1_s || 0.0);
    }
    updateScrubberAndDro();
  }

  function setLayoutMode(mode) {
    state.layoutMode = mode === "sbs" ? "sbs" : "single";
    const bSingle = document.getElementById("layout-single");
    const bSbs = document.getElementById("layout-sbs");
    if (state.layoutMode === "sbs") {
      bSbs.classList.add("btn-active");
      bSingle.classList.remove("btn-active");
    } else {
      bSingle.classList.add("btn-active");
      bSbs.classList.remove("btn-active");
    }
    refreshActivePaneAssignments(true);
  }

  function populateFeedsSpeedsDropdowns(catalog) {
    const stockMats = catalog.stock_materials || DEFAULT_FS_CATALOG.stock_materials;
    const toolMats = catalog.tool_materials || DEFAULT_FS_CATALOG.tool_materials;

    for (const selId of ["stock-mat-select", "mach-stock-mat"]) {
      const stockSel = document.getElementById(selId);
      if (!stockSel) continue;
      stockSel.replaceChildren();
      for (const m of stockMats) {
        stockSel.appendChild(
          el("option", {
            value: m.id,
            text: m.label + " (" + m.hardness_hb + " HB)",
          })
        );
      }
      stockSel.value = state.stockMaterialId;
    }

    for (const selId of ["tool-mat-select", "mach-tool-mat"]) {
      const toolSel = document.getElementById(selId);
      if (!toolSel) continue;
      toolSel.replaceChildren();
      for (const t of toolMats) {
        toolSel.appendChild(
          el("option", {
            value: t.id,
            text: t.label,
          })
        );
      }
      toolSel.value = state.toolMaterialId;
    }
  }

  function ingestToolCatalog(toolCat) {
    if (!toolCat || !Array.isArray(toolCat.tools)) return;
    state.toolCatalog = toolCat;
    state.toolLibrary = toolCat.tools;
    state.toolLibraryById = new Map();
    for (const t of toolCat.tools) {
      state.toolLibraryById.set(t.tool_id, t);
    }
    if (toolCat.starter_cribs) {
      state.starterCribs = toolCat.starter_cribs;
    }
    if (state.selectedToolIds.size === 0) {
      const defCrib =
        (state.starterCribs && state.starterCribs.us_haas_10t && state.starterCribs.us_haas_10t.tool_ids) ||
        toolCat.tools
          .filter((t) => t.unit_system === "inches" && t.is_3axis_mill_compatible !== false)
          .slice(0, 12)
          .map((t) => t.tool_id);
      state.selectedToolIds = new Set(defCrib);
    }
    updateModalCribBadge();
  }

  function populateExamplePartsSelect() {
    const sel = document.getElementById("example-part-select");
    if (!sel) return;
    sel.replaceChildren();
    const list =
      state.exampleParts.length > 0
        ? state.exampleParts
        : [
            { id: "Titan-M1", filename: "Titan-M1.step", label: "Titan-M1.step — Prismatic Plate + C-Sinks (Inch)" },
            { id: "Titan-M2", filename: "Titan-M2.step", label: "Titan-M2.step — Boss & Pocket Plate (Inch)" },
            { id: "Titan-M3", filename: "Titan-M3.step", label: "Titan-M3.step — Angled Contour Plate (Inch)" },
            { id: "Titan-M4", filename: "Titan-M4.step", label: "Titan-M4.step — Stepped Bore Housing (Inch)" },
            { id: "Titan-M5", filename: "Titan-M5.step", label: "Titan-M5.step — Multi-Pocket Bracket (Inch)" },
            { id: "Titan-M6", filename: "Titan-M6.step", label: "Titan-M6.step — Slotted Flange Plate (Inch)" },
            { id: "Titan-M7", filename: "Titan-M7.step", label: "Titan-M7.step — Precision Bearing Block (Inch)" },
            { id: "Titan-M8", filename: "Titan-M8.step", label: "Titan-M8.step — Circular Boss Manifold (Inch)" },
            { id: "Titan-M9", filename: "Titan-M9.step", label: "Titan-M9.step — 3D B-Spline Sculptured Cap (Inch)" },
            { id: "Titan-M10", filename: "Titan-M10.step", label: "Titan-M10.step — 3D Sculptured Aero Plate (Inch)" },
          ];
    for (const item of list) {
      sel.appendChild(
        el("option", {
          value: item.id || item.filename,
          text: item.label || item.filename,
        })
      );
    }
  }

  function upsertCaseOptionInDropdown(caseEntry) {
    const select = document.getElementById("case-select");
    if (!select || !caseEntry || !caseEntry.case_json_url) return;
    const gainVal = Number(
      caseEntry.chipload_gain_pct !== undefined ? caseEntry.chipload_gain_pct : caseEntry.ste_gain_pct || 0.0
    );
    const optText =
      caseEntry.case_id +
      " (" +
      (caseEntry.family || "STEP CAD") +
      " · -" +
      gainVal.toFixed(1) +
      "%)";

    for (const opt of Array.from(select.options)) {
      if (opt.value === caseEntry.case_json_url || opt.getAttribute("data-cid") === caseEntry.case_id) {
        opt.value = caseEntry.case_json_url;
        opt.textContent = optText;
        return;
      }
    }
    const newOpt = el("option", {
      value: caseEntry.case_json_url,
      "data-cid": caseEntry.case_id,
      text: optText,
    });
    select.insertBefore(newOpt, select.firstChild);
  }

  async function probeInteractiveServer() {
    try {
      const resp = await fetch("/api/session", {
        method: "GET",
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      if (!resp.ok) return;
      const sess = await resp.json();
      if (sess && typeof sess.csrf_token === "string" && sess.csrf_token.length >= 16) {
        state.csrfToken = sess.csrf_token;
        state.serverInteractive = true;
      }
      if (sess.tool_catalog) {
        ingestToolCatalog(sess.tool_catalog);
      }
      if (Array.isArray(sess.example_parts) && sess.example_parts.length > 0) {
        state.exampleParts = sess.example_parts;
      }
      populateExamplePartsSelect();
      if (Array.isArray(sess.dynamic_cases)) {
        for (const dc of sess.dynamic_cases) {
          upsertCaseOptionInDropdown(dc);
        }
      }
    } catch (_err) {
      populateExamplePartsSelect();
    }
  }

  const MAX_CLIENT_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MiB

  function setUploadModalStatus(msg, isError) {
    const box = document.getElementById("mach-status-box");
    if (!box) return;
    if (!msg) {
      box.classList.add("hidden");
      box.textContent = "";
      return;
    }
    box.className = isError ? "status-banner status-banner-error" : "status-banner";
    box.textContent = msg;
  }

  function openUploadModal() {
    const modal = document.getElementById("upload-modal");
    if (!modal) return;
    setUploadModalStatus("", false);
    updateModalCribBadge();
    const machUnit = document.getElementById("mach-unit-system");
    if (machUnit) machUnit.value = state.unitSystem;
    const machStock = document.getElementById("mach-stock-mat");
    if (machStock) machStock.value = state.stockMaterialId;
    const machTool = document.getElementById("mach-tool-mat");
    if (machTool) machTool.value = state.toolMaterialId;
    const machDial = document.getElementById("mach-dialect");
    if (machDial) machDial.value = state.gcodeDialect;
    modal.classList.remove("hidden");
  }

  function closeUploadModal() {
    const modal = document.getElementById("upload-modal");
    if (modal) modal.classList.add("hidden");
  }

  function setUploadSourceMode(mode) {
    state.uploadSourceMode = mode === "example" ? "example" : "upload";
    const bUp = document.getElementById("src-mode-upload");
    const bEx = document.getElementById("src-mode-example");
    const pUp = document.getElementById("src-upload-panel");
    const pEx = document.getElementById("src-example-panel");
    if (state.uploadSourceMode === "example") {
      bEx.classList.add("btn-active");
      bUp.classList.remove("btn-active");
      pEx.classList.remove("hidden");
      pUp.classList.add("hidden");
    } else {
      bUp.classList.add("btn-active");
      bEx.classList.remove("btn-active");
      pUp.classList.remove("hidden");
      pEx.classList.add("hidden");
    }
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    const parts = [];
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const sub = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
      parts.push(String.fromCharCode.apply(null, sub));
    }
    return btoa(parts.join(""));
  }

  async function handleCadFileSelected(file) {
    const summaryEl = document.getElementById("cad-file-summary");
    if (!file) return;
    const rawName = String(file.name || "");
    const lower = rawName.toLowerCase();
    if (!lower.endsWith(".step") && !lower.endsWith(".stp") && !lower.endsWith(".stl")) {
      setUploadModalStatus("Unsupported file extension. Only .step, .stp, and .stl solid models are permitted.", true);
      return;
    }
    if (file.size <= 0 || file.size > MAX_CLIENT_UPLOAD_BYTES) {
      setUploadModalStatus("CAD file must be non-empty and <= 10 MB.", true);
      return;
    }
    const buf = await file.arrayBuffer();
    state.pendingUploadFile = file;
    state.pendingUploadBase64 = arrayBufferToBase64(buf);
    if (summaryEl) {
      summaryEl.textContent = rawName + " (" + (file.size / 1024).toFixed(1) + " KB) ready for B-Rep planning";
    }
    const titleInput = document.getElementById("mach-part-title");
    if (titleInput && !titleInput.value.trim()) {
      titleInput.value = rawName.replace(/\.(step|stp|stl)$/i, "").slice(0, 40);
    }
    setUploadModalStatus("", false);
  }

  async function submitMachiningRequest(payload, fromModal) {
    if (state.machiningBusy) return;
    if (!state.serverInteractive || !state.csrfToken) {
      const msg = "Live CAD planning requires scripts/serve_simulator.py (active session token not detected).";
      if (fromModal) setUploadModalStatus(msg, true);
      else showError(msg);
      return;
    }

    state.machiningBusy = true;
    const runBtn = document.getElementById("btn-run-machining");
    const replanTopBtn = document.getElementById("btn-replan-current");
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.textContent = "Planning & Optimizing…";
    }
    if (replanTopBtn) {
      replanTopBtn.disabled = true;
      replanTopBtn.textContent = "Re-Planning…";
    }
    if (fromModal) {
      setUploadModalStatus(
        "Running 6-Axis B-Rep Feature Extraction → Tool Crib TAMP → Persistent-IPW DiffCAM CWE Optimization…",
        false
      );
    }

    try {
      const resp = await fetch("/api/machine-part", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          "X-CSRF-Token": state.csrfToken,
        },
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.error || ("Machining planner returned HTTP " + resp.status));
      }

      if (payload.stock_material_id) {
        state.stockMaterialId = payload.stock_material_id;
        const sm = document.getElementById("stock-mat-select");
        if (sm) sm.value = state.stockMaterialId;
      }
      if (payload.tool_material_id) {
        state.toolMaterialId = payload.tool_material_id;
        const tm = document.getElementById("tool-mat-select");
        if (tm) tm.value = state.toolMaterialId;
      }
      if (payload.controller_dialect) {
        state.gcodeDialect = payload.controller_dialect;
      }

      state.speedPolicy = "optimal_chipload";
      const spSel = document.getElementById("speed-policy-select");
      if (spSel) spSel.value = "optimal_chipload";

      if (data.case_entry) {
        upsertCaseOptionInDropdown(data.case_entry);
        const caseSel = document.getElementById("case-select");
        if (caseSel) caseSel.value = data.case_entry.case_json_url;
      }
      state.stlCache.clear();
      closeUploadModal();

      const nextUrl = (data.case_entry && data.case_entry.case_json_url) || data.case_json_url;
      if (nextUrl) {
        await loadCase(nextUrl + "?v=" + Date.now(), "chipload_ste");
      }
    } catch (err) {
      const errMsg = err.message || "Automated CAD machining failed.";
      if (fromModal) setUploadModalStatus(errMsg, true);
      else showError(errMsg);
    } finally {
      state.machiningBusy = false;
      if (runBtn) {
        runBtn.disabled = false;
        runBtn.textContent = "Plan Tools & Optimize Trajectories";
      }
      if (replanTopBtn) {
        replanTopBtn.disabled = false;
        replanTopBtn.textContent = "Re-Plan w/ Crib";
      }
    }
  }

  function triggerReplanCurrentPart() {
    if (!state.currentCase) return;
    const cribList = Array.from(state.selectedToolIds);
    if (cribList.length === 0) {
      showError("Select at least one cutting tool in the Tool Crib tab before re-planning.");
      return;
    }
    const cid = String(state.currentCase.case_id || "");
    const profEl = document.getElementById("mach-probing-profile");
    const otsEl = document.getElementById("chk-mach-ots");
    const prbEl = document.getElementById("chk-mach-probing");
    const payload = {
      source_mode: "replan_case",
      case_id: cid,
      source_cad_id: state.currentCase.source_cad_id || cid,
      part_title: state.currentCase.part_title || cid,
      unit_system: state.unitSystem,
      crib_mode: "selected_crib",
      allowed_tool_ids: cribList,
      setup_mode: "auto",
      stock_material_id: state.stockMaterialId,
      tool_material_id: state.toolMaterialId,
      controller_dialect: state.gcodeDialect,
      opt_iters: 24,
      enable_probing: prbEl ? Boolean(prbEl.checked) : true,
      probing_profile: profEl ? String(profEl.value || "haas_wips_r") : "haas_wips_r",
      enable_ots: otsEl ? Boolean(otsEl.checked) : true,
      enable_chamfer: true,
      enable_threading: true,
    };
    submitMachiningRequest(payload, false);
  }

  function bindEvents(canvas) {
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("mousedown", (e) => {
      state.cam.dragging = true;
      state.cam.panMode = e.button === 2 || e.button === 1 || e.shiftKey;
      state.cam.lastX = e.clientX;
      state.cam.lastY = e.clientY;
    });

    window.addEventListener("mouseup", () => {
      state.cam.dragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      if (!state.cam.dragging) return;
      const dx = e.clientX - state.cam.lastX;
      const dy = e.clientY - state.cam.lastY;
      state.cam.lastX = e.clientX;
      state.cam.lastY = e.clientY;

      if (state.cam.panMode) {
        const scale = state.cam.distance * 0.0016;
        const rightX = -Math.sin(state.cam.yaw);
        const rightY = Math.cos(state.cam.yaw);
        state.cam.target[0] -= dx * scale * rightX;
        state.cam.target[1] -= dx * scale * rightY;
        state.cam.target[2] += dy * scale;
      } else {
        state.cam.yaw -= dx * 0.0075;
        state.cam.pitch = Math.max(-1.48, Math.min(1.52, state.cam.pitch + dy * 0.0075));
      }
    });

    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const factor = Math.exp(e.deltaY * 0.0012);
        state.cam.distance = Math.max(18.0, Math.min(920.0, state.cam.distance * factor));
      },
      { passive: false }
    );

    document.getElementById("layout-single").addEventListener("click", () => setLayoutMode("single"));
    document.getElementById("layout-sbs").addEventListener("click", () => setLayoutMode("sbs"));

    document.getElementById("speed-policy-select").addEventListener("change", (e) => {
      const val = String(e.target.value || "matched");
      state.speedPolicy = val === "optimal_chipload" ? "optimal_chipload" : "matched";
      reapplyPhysicsPolicyToCase(true);
    });

    document.getElementById("aggressiveness-slider").addEventListener("input", (e) => {
      const pct = Math.max(40, Math.min(180, Number(e.target.value || 100)));
      state.aggressiveness = pct / 100.0;
      document.getElementById("aggressiveness-val").textContent = Math.round(pct) + "%";
      reapplyPhysicsPolicyToCase(true);
    });

    document.getElementById("stock-mat-select").addEventListener("change", (e) => {
      const allowed = (state.catalog.stock_materials || DEFAULT_FS_CATALOG.stock_materials).map((m) => m.id);
      const val = String(e.target.value || "al_6061_t6");
      if (allowed.includes(val)) {
        state.stockMaterialId = val;
        reapplyPhysicsPolicyToCase(true);
      }
    });

    document.getElementById("tool-mat-select").addEventListener("change", (e) => {
      const allowed = (state.catalog.tool_materials || DEFAULT_FS_CATALOG.tool_materials).map((t) => t.id);
      const val = String(e.target.value || "carbide_coated");
      if (allowed.includes(val)) {
        state.toolMaterialId = val;
        reapplyPhysicsPolicyToCase(true);
      }
    });

    document.getElementById("cam-iso").addEventListener("click", () => {
      state.cam.yaw = -0.78;
      state.cam.pitch = 0.62;
      state.cam.distance = defaultCamDistance(state.currentCase);
    });
    document.getElementById("cam-top").addEventListener("click", () => {
      state.cam.yaw = -Math.PI * 0.5;
      state.cam.pitch = 1.52;
    });
    document.getElementById("cam-front").addEventListener("click", () => {
      state.cam.yaw = -Math.PI * 0.5;
      state.cam.pitch = 0.06;
    });
    document.getElementById("cam-fit").addEventListener("click", () => {
      state.cam.yaw = -0.78;
      state.cam.pitch = 0.62;
      state.cam.distance = defaultCamDistance(state.currentCase);
      if (state.currentCase) {
        state.cam.target = [
          0.5 * (state.currentCase.stock_min_mm[0] + state.currentCase.stock_max_mm[0]),
          0.5 * (state.currentCase.stock_min_mm[1] + state.currentCase.stock_max_mm[1]),
          0.5 * (state.currentCase.stock_min_mm[2] + state.currentCase.stock_max_mm[2]),
        ];
      }
    });

    document.getElementById("stock-mode-select").addEventListener("change", (e) => {
      state.stockMode = String(e.target.value || "live_alloy");
      const alphaSlider = document.getElementById("stock-opacity");
      if (state.stockMode === "live_glass" && state.stockAlpha > 0.85) {
        state.stockAlpha = 0.48;
        alphaSlider.value = "50";
      } else if (state.stockMode !== "live_glass" && state.stockAlpha < 0.85) {
        state.stockAlpha = 1.0;
        alphaSlider.value = "100";
      }
      for (const slot of renderer.panes) {
        if (slot.dexelSim) slot.dexelSim.lastCarveKey = "";
      }
      updateScrubberAndDro();
      refreshHudLegend();
    });

    document.getElementById("stock-color-select").addEventListener("change", (e) => {
      state.stockPalette = String(e.target.value || "green");
      for (const slot of renderer.panes) {
        if (slot.dexelSim) slot.dexelSim.lastCarveKey = "";
      }
      updateScrubberAndDro();
    });

    document.getElementById("stock-opacity").addEventListener("input", (e) => {
      const pct = Number(e.target.value || 100);
      state.stockAlpha = Math.max(0.15, Math.min(1.0, pct / 100.0));
    });

    document.getElementById("path-color-select").addEventListener("change", (e) => {
      state.pathColorMode = String(e.target.value || "chipload");
      rebuildAllPaneToolpaths();
    });

    document.getElementById("path-width-slider").addEventListener("input", (e) => {
      const raw = Math.max(10, Math.min(50, Number(e.target.value || 28)));
      state.pathWidthMm = raw / 100.0;
      rebuildAllPaneToolpaths();
      updateScrubberAndDro();
    });

    const btnOpsAll = document.getElementById("btn-ops-all");
    if (btnOpsAll) {
      btnOpsAll.addEventListener("click", () => {
        applyOpSelectionSet(new Set(state.operations.map((o) => o.key)), false);
      });
    }
    const btnOpsG54 = document.getElementById("btn-ops-g54");
    if (btnOpsG54) {
      btnOpsG54.addEventListener("click", () => {
        const g54Keys = state.operations.filter((o) => o.setupIndex === 1).map((o) => o.key);
        applyOpSelectionSet(new Set(g54Keys), true);
      });
    }
    const btnOpsG55 = document.getElementById("btn-ops-g55");
    if (btnOpsG55) {
      btnOpsG55.addEventListener("click", () => {
        const g55Keys = state.operations.filter((o) => o.setupIndex >= 2).map((o) => o.key);
        if (g55Keys.length > 0) {
          applyOpSelectionSet(new Set(g55Keys), true);
        }
      });
    }
    const btnPrevOp = document.getElementById("btn-prev-op");
    if (btnPrevOp) {
      btnPrevOp.addEventListener("click", () => stepOperation(-1));
    }
    const btnNextOp = document.getElementById("btn-next-op");
    if (btnNextOp) {
      btnNextOp.addEventListener("click", () => stepOperation(1));
    }
    const opTimelineMount = document.getElementById("op-timeline-mount");
    if (opTimelineMount) {
      opTimelineMount.addEventListener("click", (e) => {
        const rect = opTimelineMount.getBoundingClientRect();
        if (rect.width > 4 && state.totalDurationS > 1e-6) {
          const frac = Math.max(0.0, Math.min(1.0, (e.clientX - rect.left) / rect.width));
          state.playing = false;
          document.getElementById("btn-play").textContent = "Play";
          state.playTimeS = frac * state.totalDurationS;
          updateScrubberAndDro();
        }
      });
    }

    document.getElementById("btn-play").addEventListener("click", () => {
      if (state.totalDurationS <= 1e-6) return;
      if (!state.playing) {
        if (state.opFilterMode === "custom" && state.selectedOpKeys.size > 0 && state.operations.length > 0) {
          const selOps = state.operations.filter((o) => state.selectedOpKeys.has(o.key));
          if (selOps.length > 0) {
            const firstStart = selOps[0].t0S;
            const lastEnd = selOps[selOps.length - 1].t1S;
            if (state.playTimeS >= lastEnd - 1e-4 || state.playTimeS < firstStart - 1e-4) {
              state.playTimeS = firstStart;
            }
          }
        } else if (state.playTimeS >= state.totalDurationS - 1e-4) {
          state.playTimeS = 0.0;
        }
      }
      state.playing = !state.playing;
      state.lastFrameMs = performance.now();
      document.getElementById("btn-play").textContent = state.playing ? "Pause" : "Play";
      updateScrubberAndDro();
    });

    document.getElementById("btn-reset").addEventListener("click", () => {
      state.playing = false;
      state.playTimeS = 0.0;
      document.getElementById("btn-play").textContent = "Play";
      updateScrubberAndDro();
    });

    document.getElementById("btn-prev").addEventListener("click", () => {
      stepMove(-1);
    });

    document.getElementById("btn-next").addEventListener("click", () => {
      stepMove(1);
    });

    document.getElementById("btn-end").addEventListener("click", () => {
      state.playing = false;
      state.playTimeS = state.totalDurationS;
      document.getElementById("btn-play").textContent = "Play";
      updateScrubberAndDro();
    });

    document.getElementById("speed-select").addEventListener("change", (e) => {
      const val = Number(e.target.value);
      if (Number.isFinite(val) && val > 0) {
        state.speed = val;
      }
    });

    document.getElementById("time-slider").addEventListener("input", (e) => {
      state.playing = false;
      document.getElementById("btn-play").textContent = "Play";
      const frac = Number(e.target.value) / 1000.0;
      state.playTimeS = Math.max(0.0, Math.min(1.0, frac)) * state.totalDurationS;
      updateScrubberAndDro();
    });

    document.getElementById("case-select").addEventListener("change", (e) => {
      loadCase(e.target.value);
    });

    for (const btn of Array.from(document.querySelectorAll(".tab-btn"))) {
      btn.addEventListener("click", () => {
        for (const other of Array.from(document.querySelectorAll(".tab-btn"))) {
          other.classList.remove("active");
          other.setAttribute("aria-selected", "false");
        }
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        state.activeTab = btn.getAttribute("data-tab") || "chipload";
        renderInspectorTab();
      });
    }

    // Upload / Machine CAD Modal & Re-Plan Controls
    const btnOpenUpload = document.getElementById("btn-open-upload");
    if (btnOpenUpload) btnOpenUpload.addEventListener("click", openUploadModal);
    const btnCloseUpload = document.getElementById("btn-close-upload");
    if (btnCloseUpload) btnCloseUpload.addEventListener("click", closeUploadModal);
    const btnCancelUpload = document.getElementById("btn-cancel-upload");
    if (btnCancelUpload) btnCancelUpload.addEventListener("click", closeUploadModal);
    const btnReplanTop = document.getElementById("btn-replan-current");
    if (btnReplanTop) btnReplanTop.addEventListener("click", triggerReplanCurrentPart);

    const srcUpBtn = document.getElementById("src-mode-upload");
    if (srcUpBtn) srcUpBtn.addEventListener("click", () => setUploadSourceMode("upload"));
    const srcExBtn = document.getElementById("src-mode-example");
    if (srcExBtn) srcExBtn.addEventListener("click", () => setUploadSourceMode("example"));

    const fileInput = document.getElementById("cad-file-input");
    if (fileInput) {
      fileInput.addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) handleCadFileSelected(f);
      });
    }

    const dropzone = document.querySelector(".dropzone");
    if (dropzone) {
      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
      });
      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) handleCadFileSelected(f);
      });
    }

    const machUnitSel = document.getElementById("mach-unit-system");
    if (machUnitSel) {
      machUnitSel.addEventListener("change", (e) => {
        state.unitSystem = String(e.target.value || "inches");
        const allTools = state.toolLibrary.filter((t) => t.is_3axis_mill_compatible !== false);
        const hasChecked = allTools.some(
          (t) => state.selectedToolIds.has(t.tool_id) && (state.unitSystem === "both" || t.unit_system === state.unitSystem)
        );
        if (!hasChecked) {
          applyStarterCribPreset(state.unitSystem === "millimeters" ? "metric_haas_10t" : "us_haas_10t");
        }
      });
    }

    const btnJumpCrib = document.getElementById("btn-jump-crib");
    if (btnJumpCrib) {
      btnJumpCrib.addEventListener("click", () => {
        closeUploadModal();
        for (const other of Array.from(document.querySelectorAll(".tab-btn"))) {
          const isTools = other.getAttribute("data-tab") === "tools";
          other.classList.toggle("active", isTools);
          other.setAttribute("aria-selected", isTools ? "true" : "false");
        }
        state.activeTab = "tools";
        renderInspectorTab();
      });
    }

    const btnRunMachining = document.getElementById("btn-run-machining");
    if (btnRunMachining) {
      btnRunMachining.addEventListener("click", () => {
        const cribMode = document.getElementById("mach-crib-mode").value;
        const unitSys = document.getElementById("mach-unit-system").value;
        const allowedIds = cribMode === "selected_crib" ? Array.from(state.selectedToolIds) : null;
        if (cribMode === "selected_crib" && (!allowedIds || allowedIds.length === 0)) {
          setUploadModalStatus("No tools are checked in your Tool Crib. Select tools or choose Full Unit Library.", true);
          return;
        }
        if (state.uploadSourceMode === "upload" && !state.pendingUploadBase64) {
          setUploadModalStatus("Please choose a .step, .stp, or .stl file first (or switch to Example Parts).", true);
          return;
        }
        const exSel = document.getElementById("example-part-select");
        const payload = {
          source_mode: state.uploadSourceMode,
          filename: state.pendingUploadFile ? state.pendingUploadFile.name : "",
          cad_base64: state.uploadSourceMode === "upload" ? state.pendingUploadBase64 : "",
          example_id: state.uploadSourceMode === "example" && exSel ? exSel.value : "",
          part_title: document.getElementById("mach-part-title").value.trim(),
          unit_system: unitSys,
          crib_mode: cribMode,
          allowed_tool_ids: allowedIds,
          setup_mode: document.getElementById("mach-setup-mode").value,
          stock_material_id: document.getElementById("mach-stock-mat").value,
          tool_material_id: document.getElementById("mach-tool-mat").value,
          controller_dialect: document.getElementById("mach-dialect").value,
          opt_iters: Number(document.getElementById("mach-opt-iters").value || 28),
          enable_probing: document.getElementById("chk-mach-probing").checked,
          probing_profile: document.getElementById("mach-probing-profile")
            ? document.getElementById("mach-probing-profile").value
            : "haas_wips_r",
          enable_ots: document.getElementById("chk-mach-ots")
            ? document.getElementById("chk-mach-ots").checked
            : true,
          enable_chamfer: document.getElementById("chk-mach-chamfer").checked,
          enable_threading: document.getElementById("chk-mach-thread").checked,
        };
        submitMachiningRequest(payload, true);
      });
    }
  }

  // --- Bootstrap ---
  async function boot() {
    try {
      const canvas = document.getElementById("gl-canvas");
      renderer = initWebGL(canvas);
      bindEvents(canvas);

      const manifest = await fetchJson("/data/manifest.json");
      state.manifest = manifest;
      if (manifest.feeds_and_speeds_catalog) {
        state.catalog = manifest.feeds_and_speeds_catalog;
      }
      if (manifest.tool_catalog) {
        ingestToolCatalog(manifest.tool_catalog);
      }
      if (Array.isArray(manifest.example_parts) && manifest.example_parts.length > 0) {
        state.exampleParts = manifest.example_parts;
      }
      populateFeedsSpeedsDropdowns(state.catalog);
      populateExamplePartsSelect();

      const select = document.getElementById("case-select");
      select.replaceChildren();
      for (const c of manifest.cases || []) {
        const gainVal = Number(
          c.chipload_gain_pct !== undefined ? c.chipload_gain_pct : c.ste_gain_pct || 0
        );
        select.appendChild(
          el("option", {
            value: c.case_json_url,
            "data-cid": c.case_id,
            text:
              c.case_id +
              " (" +
              c.family +
              " · -" +
              gainVal.toFixed(1) +
              "%)",
          })
        );
      }

      await probeInteractiveServer();

      if (manifest.cases && manifest.cases.length > 0) {
        const defaultCase =
          manifest.cases.find((c) => c.case_id === "titan-m1") ||
          manifest.cases.find((c) => c.case_id === "pocket-test-00") ||
          manifest.cases[0];
        select.value = defaultCase.case_json_url;
        await loadCase(defaultCase.case_json_url, "chipload_ste");
      }
      state.lastFrameMs = performance.now();
      requestAnimationFrame(renderFrame);
    } catch (err) {
      showError(err.message || "Failed to initialize DiffCAM Manufacture Simulator.");
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      DEFAULT_FS_CATALOG,
      computeFeedsAndSpeeds,
      radialChipThinningMultiplier,
      axialChipThinningMultiplier,
      blendedSegmentDurations,
      computeTightColorRange,
      buildThickPathTubeGeometry,
      buildToolAssemblyMesh,
      resolveCaseWorkholding,
      resolveCaseSetupSheet,
      formatSetupSheetMarkdown,
      buildKurtDx6ViseMesh,
      deriveCaseOperations,
      cutSegmentColorRgb,
      mat4FlipX180,
    };
  }

  if (typeof window !== "undefined" && typeof document !== "undefined") {
    window.addEventListener("DOMContentLoaded", boot);
  }
})();

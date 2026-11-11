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

  function buildCutterSpecFromCase(caseData) {
    const t = (caseData && caseData.tool) || {};
    const diam = Number(t.diameter_mm || (t.cutter_radius_mm ? 2.0 * t.cutter_radius_mm : 5.0));
    const fluteLen = Number(t.flute_length_mm || 12.0);
    const stickout = Math.max(fluteLen + 2.0, Number(t.holder_bottom_mm || 14.0));
    const shankDiam = Number(t.shank_radius_mm ? 2.0 * t.shank_radius_mm : diam);
    return {
      diameter_mm: diam,
      flutes: 3,
      flute_length_mm: fluteLen,
      stickout_mm: stickout,
      shank_diameter_mm: shankDiam,
      corner_radius_mm: 0.0,
      helix_deg: 35.0,
      slot_ipt_factor: 0.80,
      tool_type_ipt: 1.0,
      max_rpm: 24000.0,
      max_deflection_pct: 100.0,
      max_torque_pct: 100.0,
    };
  }

  // Re-evaluate a variant's segment speeds, durations, and chipload under current Fair / Optimal Chipload policy
  function evaluateVariantUnderPolicy(variant, caseData, settings, baseRes, catalog) {
    const origPlan = variant.origPlan || variant.plan;
    if (!variant.origPlan) {
      variant.origPlan = JSON.parse(JSON.stringify(variant.plan));
    }
    const srcSegs = variant.origPlan.segments || [];
    const cutter = buildCutterSpecFromCase(caseData);
    const rapidSpeed = Number(caseData.rapid_mm_s || 120.0);
    const maxAccel = Number(caseData.max_accel_mm_s2 || 1500.0);
    const nomDoc = Number(caseData.nominal_doc_mm || 3.0);
    const nomWoc = Number(caseData.nominal_woc_mm || 1.75);
    const baseFeedS = Math.max(1.0, Number(baseRes.feed_mm_s));
    const spindleRpm = Math.max(100.0, Number(baseRes.spindle_rpm));
    const toothFreqHz = (spindleRpm * cutter.flutes) / 60.0;
    const hOptTarget = Math.max(0.004, Number(baseRes.chip_thickness_mm));
    const hBandMin = 0.70 * hOptTarget;
    const hBandMax = 1.25 * hOptTarget;

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
          cutter,
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
      const isAtcDwell = src.kind === "rapid" && Number(src.length_mm || 0.0) <= 1e-6 && Number(src.duration_s || 0.0) >= 0.5;
      const dur = isAtcDwell ? Number(src.duration_s) : Math.max(1e-7, Number(durs[i]));
      const spd = Number(speeds[i]);
      const isRapid = src.kind === "rapid";

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
          const uAe = Math.max(0.0, Math.min(1.0, Number(src.ae_mm || 0.0) / Math.max(cutter.diameter_mm, 1e-6)));
          const psi = radialChipThinningPsi(uAe);
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
        tool_radius_mm: Number(src.tool_radius_mm || (cutter.diameter_mm * 0.5)),
        op_id: src.op_id || "",
      };
      tCursor += dur;
    }

    const totalT = Math.max(tCursor, 1e-9);
    const denomEng = Math.max(totalEngTime, 1e-9);
    const origEng = origPlan.engagement_summary || {};

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
        target_hex_mm: Number(hOptTarget.toFixed(5)),
        optimal_hex_band_mm: [Number(hBandMin.toFixed(5)), Number(hBandMax.toFixed(5))],
        optimal_hex_pct: totalEngTime > 1e-6 ? Number(((100.0 * optHexTime) / denomEng).toFixed(2)) : 0.0,
        peak_inst_mrr_mm3_s: Number(peakMrr.toFixed(2)),
        spindle_rpm: spindleRpm,
      }),
    });

    if (!variant.origMetrics) {
      variant.origMetrics = Object.assign({}, variant.metrics);
    }
    variant.metrics.cycle_time_s = Number(tCursor.toFixed(3));
    variant.metrics.peak_mrr_mm3_s = Number(peakMrr.toFixed(2));
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

  // --- Clean-Room Real-Time 2.5D Z-Dexel Cutting Stock Simulator ---
  class DexelStockSim {
    constructor(caseData, pitchMm) {
      this.pitch = pitchMm || DEXEL_PITCH_MM;
      this.smin = caseData.stock_min_mm.map(Number);
      this.smax = caseData.stock_max_mm.map(Number);
      this.rCut = Number((caseData.tool && caseData.tool.cutter_radius_mm) || 2.5);
      this.r2 = this.rCut * this.rCut;

      const lx = Math.max(2.0, this.smax[0] - this.smin[0]);
      const ly = Math.max(2.0, this.smax[1] - this.smin[1]);
      this.nx = Math.max(8, Math.round(lx / this.pitch));
      this.ny = Math.max(8, Math.round(ly / this.pitch));
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
      this.lastCutSeg = new Int32Array(count);

      const maxTris = (3 * count + 2 * (this.nx + this.ny) + 4) * 2;
      this.posScratch = new Float32Array(maxTris * 9);
      this.nrmScratch = new Float32Array(maxTris * 9);
      this.colScratch = new Float32Array(maxTris * 9);

      this.segments = [];
      this.keyframes = [];
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

    setSegments(segments) {
      this.segments = Array.isArray(segments) ? segments : [];
      this.keyframes = [];
      this.keyframeCutSeg = [];

      const zCur = new Float32Array(this.nx * this.ny);
      zCur.fill(this.smax[2]);
      const cutSeg = new Int32Array(this.nx * this.ny);
      cutSeg.fill(-999);

      this.keyframes.push(zCur.slice());
      this.keyframeCutSeg.push(cutSeg.slice());

      for (let k = 0; k < this.segments.length; k++) {
        const s = this.segments[k];
        if (s.kind !== "rapid") {
          this._sweepLinear(
            zCur,
            cutSeg,
            k,
            s.start[0],
            s.start[1],
            s.start[2],
            s.end[0],
            s.end[1],
            s.end[2],
            s.tool_radius_mm
          );
        }
        if ((k + 1) % KEYFRAME_STRIDE === 0) {
          this.keyframes.push(zCur.slice());
          this.keyframeCutSeg.push(cutSeg.slice());
        }
      }
      this.lastCarveKey = "";
    }

    _sweepLinear(zGrid, cutSeg, segIdx, ax, ay, az, bx, by, bz, rOverride) {
      const zMinSeg = Math.min(az, bz);
      if (zMinSeg >= this.smax[2] - 1e-6) return;

      const r = rOverride && Number(rOverride) > 0.1 ? Number(rOverride) : this.rCut;
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
        const zCut = Math.max(zFloor, zMinSeg);
        for (let ix = ix0; ix <= ix1; ix++) {
          const dpx = this.xs[ix] - ax;
          const dpx2 = dpx * dpx;
          const rowOff = ix * this.ny;
          for (let iy = iy0; iy <= iy1; iy++) {
            const dpy = this.ys[iy] - ay;
            if (dpx2 + dpy * dpy <= r2Local) {
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

          const uBest = dz < 0.0 ? uHi : uLo;
          const zCut = Math.max(zFloor, az + uBest * dz);
          const idx = rowOff + iy;
          if (zCut < zGrid[idx] - 1e-6) {
            zGrid[idx] = zCut;
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
      this.lastCutSeg.set(this.keyframeCutSeg[kfIndex]);

      const replayStart = kfIndex * KEYFRAME_STRIDE;
      for (let k = replayStart; k < completedCount; k++) {
        const s = this.segments[k];
        if (s.kind !== "rapid") {
          this._sweepLinear(
            this.zWork,
            this.lastCutSeg,
            k,
            s.start[0],
            s.start[1],
            s.start[2],
            s.end[0],
            s.end[1],
            s.end[2],
            s.tool_radius_mm
          );
        }
      }

      if (clampedSeg >= 0 && alpha > 1e-4) {
        const cur = this.segments[clampedSeg];
        if (cur && cur.kind !== "rapid") {
          const ex = cur.start[0] + (cur.end[0] - cur.start[0]) * alpha;
          const ey = cur.start[1] + (cur.end[1] - cur.start[1]) * alpha;
          const ez = cur.start[2] + (cur.end[2] - cur.start[2]) * alpha;
          this._sweepLinear(
            this.zWork,
            this.lastCutSeg,
            clampedSeg,
            cur.start[0],
            cur.start[1],
            cur.start[2],
            ex,
            ey,
            ez,
            cur.tool_radius_mm
          );
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
      const wallCutRgb = stockMode === "live_glass" ? pal.skin : pal.wall;

      for (let ix = 0; ix < nx; ix++) {
        const x0 = bx0 + ix * dx;
        const x1 = x0 + dx;
        const rowOff = ix * ny;
        for (let iy = 0; iy < ny; iy++) {
          const idx = rowOff + iy;
          const zTop = z[idx];
          const y0 = by0 + iy * dy;
          const y1 = y0 + dy;
          const cTop = this._cellTopColor(idx, zTop, activeSegIdx, stockMode, pal);
          emitQuad(x0, y0, zTop, x1, y0, zTop, x1, y1, zTop, x0, y1, zTop, 0, 0, 1, cTop);
        }
      }

      for (let iy = 0; iy < ny; iy++) {
        const y0 = by0 + iy * dy;
        const y1 = y0 + dy;
        const zLeft = z[iy];
        if (zLeft > bz0 + 1e-5) {
          emitQuad(bx0, y0, bz0, bx0, y0, zLeft, bx0, y1, zLeft, bx0, y1, bz0, -1, 0, 0, pal.skin);
        }
        const zRight = z[(nx - 1) * ny + iy];
        if (zRight > bz0 + 1e-5) {
          emitQuad(bx1, y0, bz0, bx1, y1, bz0, bx1, y1, zRight, bx1, y0, zRight, 1, 0, 0, pal.skin);
        }
      }

      for (let ix = 0; ix < nx - 1; ix++) {
        const xEdge = bx0 + (ix + 1) * dx;
        const offA = ix * ny;
        const offB = (ix + 1) * ny;
        for (let iy = 0; iy < ny; iy++) {
          const zA = z[offA + iy];
          const zB = z[offB + iy];
          if (Math.abs(zA - zB) <= 1e-4) continue;
          const y0 = by0 + iy * dy;
          const y1 = y0 + dy;
          if (zA > zB) {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(offA + iy, zA, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(xEdge, y0, zB, xEdge, y1, zB, xEdge, y1, zA, xEdge, y0, zA, 1, 0, 0, cStep);
          } else {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(offB + iy, zB, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(xEdge, y0, zA, xEdge, y0, zB, xEdge, y1, zB, xEdge, y1, zA, -1, 0, 0, cStep);
          }
        }
      }

      for (let ix = 0; ix < nx; ix++) {
        const x0 = bx0 + ix * dx;
        const x1 = x0 + dx;
        const rowOff = ix * ny;
        const zFront = z[rowOff];
        if (zFront > bz0 + 1e-5) {
          emitQuad(x0, by0, bz0, x1, by0, bz0, x1, by0, zFront, x0, by0, zFront, 0, -1, 0, pal.skin);
        }
        const zBack = z[rowOff + ny - 1];
        if (zBack > bz0 + 1e-5) {
          emitQuad(x0, by1, bz0, x0, by1, zBack, x1, by1, zBack, x1, by1, bz0, 0, 1, 0, pal.skin);
        }

        for (let iy = 0; iy < ny - 1; iy++) {
          const zA = z[rowOff + iy];
          const zB = z[rowOff + iy + 1];
          if (Math.abs(zA - zB) <= 1e-4) continue;
          const yEdge = by0 + (iy + 1) * dy;
          if (zA > zB) {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(rowOff + iy, zA, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(x0, yEdge, zB, x0, yEdge, zA, x1, yEdge, zA, x1, yEdge, zB, 0, 1, 0, cStep);
          } else {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(rowOff + iy + 1, zB, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(x0, yEdge, zA, x1, yEdge, zA, x1, yEdge, zB, x0, yEdge, zB, 0, -1, 0, cStep);
          }
        }
      }

      emitQuad(bx0, by0, bz0, bx0, by1, bz0, bx1, by1, bz0, bx1, by0, bz0, 0, 0, -1, pal.skin);

      return {
        positions: pos.subarray(0, o),
        normals: nrm.subarray(0, o),
        colors: col.subarray(0, o),
        vertexCount: o / 3,
      };
    }
  }

  // --- Procedural 3D Cutter + Shank + CAT40 Holder Mesh (tip at origin) ---
  function buildToolAssemblyMesh(toolSpec) {
    const rCut = Number(toolSpec.cutter_radius_mm || 2.5);
    const fluteH = Number(toolSpec.flute_length_mm || 12.0);
    const rShank = Number(toolSpec.shank_radius_mm || rCut);
    const holderBot = Math.max(Number(toolSpec.holder_bottom_mm || 14.0), fluteH + 1.5);
    const rHolder = Number(toolSpec.holder_radius_mm || 14.0);
    const holderH = Math.min(Number(toolSpec.holder_height_mm || 24.0), 26.0);

    const sections = [
      { z0: 0.0, z1: fluteH, r0: rCut, r1: rCut, color: [0.96, 0.76, 0.24] },
      { z0: fluteH, z1: holderBot, r0: rShank, r1: rShank, color: [0.68, 0.73, 0.79] },
      { z0: holderBot, z1: holderBot + holderH * 0.35, r0: rShank * 1.6, r1: rHolder, color: [0.27, 0.31, 0.37] },
      { z0: holderBot + holderH * 0.35, z1: holderBot + holderH, r0: rHolder, r1: rHolder, color: [0.22, 0.25, 0.31] },
    ];

    const pos = [];
    const nrm = [];
    const col = [];
    const slices = 28;

    function pushTri(p0, p1, p2, n0, n1, n2, c) {
      pos.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      nrm.push(n0[0], n0[1], n0[2], n1[0], n1[1], n1[2], n2[0], n2[1], n2[2]);
      col.push(c[0], c[1], c[2], c[0], c[1], c[2], c[0], c[1], c[2]);
    }

    for (const sec of sections) {
      for (let i = 0; i < slices; i++) {
        const a0 = (i / slices) * Math.PI * 2.0;
        const a1 = ((i + 1) / slices) * Math.PI * 2.0;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);

        const p00 = [sec.r0 * c0, sec.r0 * s0, sec.z0];
        const p10 = [sec.r0 * c1, sec.r0 * s1, sec.z0];
        const p11 = [sec.r1 * c1, sec.r1 * s1, sec.z1];
        const p01 = [sec.r1 * c0, sec.r1 * s0, sec.z1];
        const n0 = [c0, s0, 0.0];
        const n1 = [c1, s1, 0.0];

        pushTri(p00, p10, p11, n0, n1, n1, sec.color);
        pushTri(p00, p11, p01, n0, n1, n0, sec.color);
      }
    }
    for (let i = 0; i < slices; i++) {
      const a0 = (i / slices) * Math.PI * 2.0;
      const a1 = ((i + 1) / slices) * Math.PI * 2.0;
      pushTri(
        [0, 0, 0],
        [rCut * Math.cos(a1), rCut * Math.sin(a1), 0],
        [rCut * Math.cos(a0), rCut * Math.sin(a0), 0],
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1],
        [0.96, 0.76, 0.24]
      );
    }

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
    if (mode === "engaged_air" || mode === "tool_number") {
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
    [0.145, 0.520, 0.960], // T1: Cobalt Blue (Rough / Face Endmill)
    [0.133, 0.800, 0.420], // T2: Emerald Green (Rest-Finish Endmill)
    [0.980, 0.780, 0.120], // T3: Vivid Gold (Spot Drill)
    [0.940, 0.340, 0.560], // T4: Crimson Rose (Peck Twist Drill)
  ];

  function cutSegmentColorRgb(seg, mode, rangeObj) {
    if (mode === "engaged_air") {
      return seg.engaged ? [0.18, 0.85, 0.45] : [0.94, 0.39, 0.58];
    }
    if (mode === "tool_number") {
      const tIdx = Math.max(0, (Number(seg.tool_number || 1) - 1) % TOOL_PALETTE_RGB.length);
      return TOOL_PALETTE_RGB[tIdx];
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

  // Extrude 3D segments into 4-sided 3D prism tubes (24 vertices / segment) so path width scales cleanly on all GPUs
  function buildThickPathTubeGeometry(segments, wantRapid, radiusMm, colorFn) {
    const matchedIndices = [];
    for (let i = 0; i < segments.length; i++) {
      const isR = segments[i].kind === "rapid";
      if (isR === wantRapid) {
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
    let o = 0;

    for (let m = 0; m < nSegs; m++) {
      const segIdx = matchedIndices[m];
      const s = segments[segIdx];
      const ax = s.start[0], ay = s.start[1], az = s.start[2];
      const bx = s.end[0], by = s.end[1], bz = s.end[2];
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

  // --- Build Static Grid + G54 Triad + Billet Wireframe Lines ---
  function buildSceneGuideLines(stockMin, stockMax) {
    const verts = [];
    const cols = [];
    function addSeg(a, b, rgba) {
      verts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      cols.push(rgba[0], rgba[1], rgba[2], rgba[3], rgba[0], rgba[1], rgba[2], rgba[3]);
    }

    const zFloor = stockMin[2] - 0.05;
    const gridExtent = 50;
    for (let g = -gridExtent; g <= gridExtent; g += 10) {
      const c = g === 0 ? [0.26, 0.33, 0.43, 0.65] : [0.18, 0.23, 0.30, 0.42];
      addSeg([-gridExtent, g, zFloor], [gridExtent, g, zFloor], c);
      addSeg([g, -gridExtent, zFloor], [g, gridExtent, zFloor], c);
    }

    const axisLen = 10.0;
    addSeg([0, 0, 0.08], [axisLen, 0, 0.08], [1.0, 0.42, 0.42, 1.0]);
    addSeg([0, 0, 0.08], [0, axisLen, 0.08], [0.45, 0.92, 0.52, 1.0]);
    addSeg([0, 0, 0.08], [0, 0, axisLen], [0.42, 0.72, 1.0, 1.0]);

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

    return {
      positions: new Float32Array(verts),
      colors: new Float32Array(cols),
      gridCount: gridVertexCount,
      billetOffset: gridVertexCount,
      billetCount: verts.length / 3 - gridVertexCount,
    };
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
      },
      panes: [createPaneSlot(gl), createPaneSlot(gl), createPaneSlot(gl)],
    };
  }

  // --- Application State ---
  const state = {
    manifest: null,
    catalog: DEFAULT_FS_CATALOG,
    currentCase: null,
    currentVariant: null,
    activeTab: "chipload",
    stlCache: new Map(),
    loadSeq: 0,

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
    guideGridCount: 0,
    guideBilletOffset: 0,
    guideBilletCount: 0,

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
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) {
      throw new Error("Failed to load JSON asset (" + resp.status + ").");
    }
    return resp.json();
  }

  async function fetchParsedStl(url, tintRgb) {
    if (state.stlCache.has(url)) {
      return state.stlCache.get(url);
    }
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error("Failed to load binary STL mesh (" + resp.status + ").");
    }
    const buf = await resp.arrayBuffer();
    const parsed = parseBinaryStl(buf, tintRgb);
    state.stlCache.set(url, parsed);
    return parsed;
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
      titleEl.textContent = "Multi-Tool TAMP Sequence (T1 → T4)";
      barEl.className = "legend-gradient mode-tight";
      minEl.textContent = "T1 Rough/Face";
      midEl.textContent = "T2 Rest / T3 Spot";
      maxEl.textContent = "T4 Peck Drill";
      subEl.textContent = "Automated StandardToolLibrary + Persistent-IPW TAMP";
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

    for (let p = 0; p < nPanes; p++) {
      const slot = renderer.panes[p];
      if (!slot.variant || !slot.variant.plan) continue;
      const segs = slot.variant.plan.segments || [];

      const cutGeom = buildThickPathTubeGeometry(segs, false, cutRadius, (s) =>
        cutSegmentColorRgb(s, state.pathColorMode, state.tightRange)
      );
      uploadBuffer(gl, slot.cutPos, cutGeom.positions, false);
      uploadBuffer(gl, slot.cutNrm, cutGeom.normals, false);
      uploadBuffer(gl, slot.cutCol, cutGeom.colors, false);
      slot.cutSegIndices = cutGeom.segIndices;
      slot.cutVertsPerSeg = cutGeom.vertsPerSeg;
      slot.cutTotalVerts = cutGeom.totalVerts;

      const rapGeom = buildThickPathTubeGeometry(segs, true, rapidRadius, () => [0.96, 0.65, 0.14]);
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
      // In Side-by-Side mode, inspector highlights Direct DiffCAM vs Naive baseline
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

    rebuildAllPaneToolpaths();
    updateScrubberAndDro();
    renderKpiStrip();
    renderInspectorTab();
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

    syncVariantButtonsLabels();
    refreshActivePaneAssignments(preserveTimeFrac);
  }

  // --- Evaluate Pose & Live Dexel Carve for a Single Pane Slot at time t ---
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
      const activeTube = buildThickPathTubeGeometry(
        [activeSeg],
        activeSeg.kind === "rapid",
        state.pathWidthMm * 1.45,
        () => [1.0, 1.0, 1.0]
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
                ["Eng/Air: " + Number(eng.engaged_time_s || 0).toFixed(1) + "/" + Number(eng.air_cut_time_s || 0).toFixed(1) + "s"]
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

  // --- Update Timeline Scrubber, Step Active Pane(s), & Refresh DRO / Overlay ---
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
    syncVariantButtonsLabels();

    document.getElementById("dl-rustcnc").setAttribute("href", state.currentCase.rustcnc_download_url);
    document.getElementById("dl-camotics").setAttribute("href", variant.camotics_project_url);
    const dlNc = document.getElementById("dl-nc");
    if (variant.haas_nc_url) {
      dlNc.setAttribute("href", variant.haas_nc_url);
      dlNc.textContent = "Haas NGC (.nc)";
    } else {
      dlNc.setAttribute("href", variant.camotics_nc_url);
      dlNc.textContent = "RS-274 (.nc)";
    }

    refreshActivePaneAssignments(preservePlayhead);
  }

  // --- Load Benchmark Case ---
  async function loadCase(caseJsonUrl, preferredVariantId) {
    const mySeq = ++state.loadSeq;
    try {
      clearError();
      const caseData = await fetchJson(caseJsonUrl);
      if (mySeq !== state.loadSeq) return;
      state.currentCase = caseData;

      for (let i = 0; i < 3; i++) {
        renderer.panes[i].dexelSim = new DexelStockSim(caseData, DEXEL_PITCH_MM);
      }

      const gl = renderer.gl;
      const guides = buildSceneGuideLines(caseData.stock_min_mm, caseData.stock_max_mm);
      uploadBuffer(gl, renderer.shared.guidePos, guides.positions, false);
      uploadBuffer(gl, renderer.shared.guideCol, guides.colors, false);
      state.guideGridCount = guides.gridCount;
      state.guideBilletOffset = guides.billetOffset;
      state.guideBilletCount = guides.billetCount;

      const toolMesh = buildToolAssemblyMesh(caseData.tool || {});
      uploadBuffer(gl, renderer.shared.toolPos, toolMesh.positions, false);
      uploadBuffer(gl, renderer.shared.toolNrm, toolMesh.normals, false);
      uploadBuffer(gl, renderer.shared.toolCol, toolMesh.colors, false);
      state.toolCount = toolMesh.vertexCount;

      const targetMesh = await fetchParsedStl(caseData.target_stl_url, [0.18, 0.85, 0.64]);
      if (mySeq !== state.loadSeq) return;
      uploadBuffer(gl, renderer.shared.targetPos, targetMesh.positions, false);
      uploadBuffer(gl, renderer.shared.targetNrm, targetMesh.normals, false);
      state.targetCount = targetMesh.vertexCount;

      const cx = 0.5 * (caseData.stock_min_mm[0] + caseData.stock_max_mm[0]);
      const cy = 0.5 * (caseData.stock_min_mm[1] + caseData.stock_max_mm[1]);
      const cz = 0.5 * (caseData.stock_min_mm[2] + caseData.stock_max_mm[2]);
      state.cam.target = [cx, cy, cz];

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

  function renderInspectorTab() {
    const container = document.getElementById("tab-content");
    container.replaceChildren();
    if (!state.currentCase || !state.manifest) return;

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

      // Live dynamically recomputed table across Naive, RL, and Direct DiffCAM under active slider/material!
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

    if (state.activeTab === "case") {
      container.appendChild(
        el("div", {
          className: "section-heading",
          text: state.currentCase.case_id + " (" + state.currentCase.family + ") — 3-Seed Benchmark Summary",
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

      container.appendChild(
        el("div", { className: "section-heading", text: "Posted G-Code & CAMotics Verification" })
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
    if (state.activeTab === "exp2") {
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
      return;
    }

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

    const proj = mat4Perspective((40.0 * Math.PI) / 180.0, pw / Math.max(ph, 1), 0.5, 900.0);
    const vp = mat4Multiply(proj, view);
    const ident = mat4Identity();

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

    // 1. Solid Target CAD underneath in Glass mode
    if (state.stockMode === "live_glass" && state.targetCount > 0) {
      drawMeshBuffer(
        renderer.shared.targetPos,
        renderer.shared.targetNrm,
        null,
        state.targetCount,
        vp,
        ident,
        eye,
        [0.72, 0.80, 0.88],
        false,
        1.0,
        0.0
      );
    }

    // 2. Opaque Live Cutting Stock
    if (flags.showStock && stockOpaque && sCnt > 0) {
      drawMeshBuffer(sPos, sNrm, sCol, sCnt, vp, ident, eye, [0.72, 0.78, 0.85], true, 1.0, 0.0);
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
        ident,
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
        ident,
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
        ident,
        eye,
        [1.0, 1.0, 1.0],
        true,
        1.0,
        0.92
      );
    }

    // 4. Animated 3D Endmill + Shank + CAT40 Holder
    if (flags.showTool && state.toolCount > 0) {
      const toolModel = mat4Translation(slot.toolPos[0], slot.toolPos[1], slot.toolPos[2]);
      drawMeshBuffer(
        renderer.shared.toolPos,
        renderer.shared.toolNrm,
        renderer.shared.toolCol,
        state.toolCount,
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
        ident,
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
        ident,
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
      state.playTimeS += dt * state.speed;
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

    const flags = {
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
    const stockSel = document.getElementById("stock-mat-select");
    const toolSel = document.getElementById("tool-mat-select");
    stockSel.replaceChildren();
    for (const m of catalog.stock_materials || DEFAULT_FS_CATALOG.stock_materials) {
      stockSel.appendChild(
        el("option", {
          value: m.id,
          text: m.label + " (" + m.hardness_hb + " HB)",
        })
      );
    }
    stockSel.value = state.stockMaterialId;

    toolSel.replaceChildren();
    for (const t of catalog.tool_materials || DEFAULT_FS_CATALOG.tool_materials) {
      toolSel.appendChild(
        el("option", {
          value: t.id,
          text: t.label,
        })
      );
    }
    toolSel.value = state.toolMaterialId;
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
        state.cam.distance = Math.max(18.0, Math.min(420.0, state.cam.distance * factor));
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
      state.cam.distance = 108.0;
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
      state.cam.distance = 108.0;
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

    document.getElementById("btn-play").addEventListener("click", () => {
      if (state.totalDurationS <= 1e-6) return;
      if (!state.playing && state.playTimeS >= state.totalDurationS - 1e-4) {
        state.playTimeS = 0.0;
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
      populateFeedsSpeedsDropdowns(state.catalog);

      const select = document.getElementById("case-select");
      select.replaceChildren();
      for (const c of manifest.cases || []) {
        const gainVal = Number(
          c.chipload_gain_pct !== undefined ? c.chipload_gain_pct : c.ste_gain_pct || 0
        );
        select.appendChild(
          el("option", {
            value: c.case_json_url,
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

      if (manifest.cases && manifest.cases.length > 0) {
        const pocketDefault =
          manifest.cases.find((c) => c.case_id === "pocket-test-00") || manifest.cases[0];
        select.value = pocketDefault.case_json_url;
        await loadCase(pocketDefault.case_json_url, "chipload_ste");
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
    };
  }

  if (typeof window !== "undefined" && typeof document !== "undefined") {
    window.addEventListener("DOMContentLoaded", boot);
  }
})();

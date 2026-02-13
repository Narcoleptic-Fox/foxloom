#[derive(Debug, Clone, Copy)]
pub struct DecayConfig {
    pub importance_floor: f32,
    pub default_half_life_hours: Option<u32>,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            importance_floor: 0.05,
            default_half_life_hours: None,
        }
    }
}

pub fn decayed_importance(
    base_importance: f32,
    decay_half_life_hours: Option<u32>,
    age_hours: f32,
    cfg: &DecayConfig,
) -> f32 {
    let clamped_base = base_importance.clamp(0.0, 1.0);
    let clamped_age = age_hours.max(0.0);
    let half_life = decay_half_life_hours.or(cfg.default_half_life_hours);

    let decayed = match half_life {
        None => clamped_base,
        Some(0) => cfg.importance_floor,
        Some(h) => {
            let factor = 2f32.powf(-clamped_age / h as f32);
            clamped_base * factor
        }
    };

    decayed.clamp(cfg.importance_floor, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_life_decay_is_stable() {
        let cfg = DecayConfig {
            importance_floor: 0.05,
            default_half_life_hours: None,
        };
        let base = 0.8f32;
        let at_zero = decayed_importance(base, Some(24), 0.0, &cfg);
        let at_half = decayed_importance(base, Some(24), 24.0, &cfg);
        let at_two_half = decayed_importance(base, Some(24), 48.0, &cfg);
        assert!((at_zero - 0.8).abs() < 1e-6);
        assert!((at_half - 0.4).abs() < 1e-4);
        assert!((at_two_half - 0.2).abs() < 1e-4);
    }

    #[test]
    fn floor_and_default_half_life_work() {
        let cfg = DecayConfig {
            importance_floor: 0.1,
            default_half_life_hours: Some(12),
        };
        let no_override = decayed_importance(0.5, None, 120.0, &cfg);
        assert!((no_override - 0.1).abs() < 1e-6);

        let zero_half = decayed_importance(0.9, Some(0), 1.0, &cfg);
        assert!((zero_half - 0.1).abs() < 1e-6);
    }
}
